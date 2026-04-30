import transformers
from unsloth import FastLanguageModel  
import os
import json
import fitz
import pandas as pd
import torch
import shutil
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./adapter_checkpoint"
TOP_K_CHUNKS = 3

def extract_and_retrieve_evidence(pdf_path, sentence, top_k=TOP_K_CHUNKS):
    if not os.path.exists(pdf_path):
        return "無法找到對應的論文檔案。"
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        raw_chunks = full_text.split('\n\n')
        chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 100]
        if not chunks: return full_text[:2000]
        vectorizer = TfidfVectorizer(stop_words='english').fit(chunks + [sentence])
        chunk_vectors = vectorizer.transform(chunks)
        query_vector = vectorizer.transform([sentence])
        sim_scores = cosine_similarity(query_vector, chunk_vectors)[0]
        top_indices = sim_scores.argsort()[-top_k:][::-1]
        relevant_evidence = "\n...\n".join([chunks[i] for i in top_indices])
        return relevant_evidence[:4000] 
    except Exception as e:
        return f"PDF 解析錯誤: {str(e)}"

def main():
    if os.path.exists("/tmp/unsloth_compiled_cache"): shutil.rmtree("/tmp/unsloth_compiled_cache")
    if os.path.exists("unsloth_compiled_cache"): shutil.rmtree("unsloth_compiled_cache")

    print("載入模型與分詞器 (GPU Mode)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=True
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("設定最佳 QLoRA 參數 (r=32, alpha=32)...")
    model = FastLanguageModel.get_peft_model(
        model, 
        r=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32, 
        lora_dropout=0, 
        bias="none", 
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )

    print("讀取 classes.json 與設定標籤對應...")
    with open('classes.json', 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
    id_to_label = {int(item['id']): item['concept'] for item in classes_info}

    def safe_map_label(val):
        try:
            val_int = int(val)
            return id_to_label.get(val_int, val)
        except:
            return val

    print("準備訓練資料與 PDF 解析...")
    train_df = pd.read_csv("train.csv")
    train_df['label'] = train_df['label'].apply(safe_map_label)
    train_df['evidence_text'] = train_df.apply(lambda row: extract_and_retrieve_evidence(f"paper_evidence/{row['paper_id']}.pdf", row['text']), axis=1)

    print("處理類別不平衡 (Oversampling)...")
    max_size = train_df['label'].value_counts().max()
    resampled_dfs = [train_df]
    for class_index, group in train_df.groupby('label'):
        if len(group) < max_size: resampled_dfs.append(group.sample(max_size - len(group), replace=True))
    train_df = pd.concat(resampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    prompt_template = """<|im_start|>system
你是一個嚴謹的學術論文評審輔助系統。請仔細比對以下「評審句子」與「論文段落」，判斷該評審句子屬於哪一種幻覺類型。
可選的幻覺類型包含：Attribution Failure, Entity, Number, Overgeneralization, Temporal。
<|im_end|>
<|im_start|>user
論文段落：
{evidence}

評審句子：
{sentence}
<|im_end|>
<|im_start|>assistant
### Answer:
{label}<|im_end|>"""

    def format_prompts(examples):
        texts = [prompt_template.format(sentence=s, evidence=e, label=l) 
                 for s, e, l in zip(examples["text"], examples["evidence_text"], examples["label"])]
        return { "formatted_text": texts }

    train_dataset = Dataset.from_pandas(train_df).map(format_prompts, batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3, # 最佳 Epoch
        learning_rate=2e-4, # 最佳 LR
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        average_tokens_across_devices=False,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="formatted_text",
        dataset_num_proc=1,
        packing=False,
    )
    training_args.push_to_hub_token = None

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator, 
        tokenizer=tokenizer, 
    )
    
    print("  [Train] 開始最終正式微調...")
    trainer.train()

    print(f"  [Save] 儲存最終模型至 {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("訓練完成！請接著執行 inference.py 產生 Kaggle 繳交檔。")

if __name__ == "__main__":
    main()