import json
import pandas as pd
import os
import re
import nltk
import torch
from tqdm import tqdm
from datasets import Dataset
import torch.nn as nn
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from unsloth import FastLanguageModel  
from sentence_transformers import SentenceTransformer, util
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./checkpoint"
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
TOP_K = 3
CHUNK_SIZE = 100
OVERLAP = 10
CHUNK_POOL = {}
BM25_POOL = {}
# make directory for results if not exists
if not os.path.exists("./results"):
    os.makedirs("./results")
CHUNKS_OUTPUT = "./results/chunks.json"
EVIDENCE_OUTPUT = "./results/train_evidence.pkl"
EMBED_MODEL = SentenceTransformer('Snowflake/snowflake-arctic-embed-l-v2.0', device=DEVICE)

def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    # 論文通常 .!? 是可靠分句點
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def build_sentence_chunks(sentences, max_words=120, overlap=2):
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent.split())
        # 如果超過 chunk size → 先切
        if current_len + sent_len > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # overlap：保留最後幾句（語意連續）
                current_chunk = current_chunk[-overlap:]
                current_len = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sent)
        current_len += sent_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def process_pdf_to_chunks(pdf_path, paper_id):
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            try:
                pages_text.append(page.get_text("text"))
            except:
                continue
        # 保留 page structure（很重要）
        full_text = "\n".join(pages_text) 
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        if not full_text:
            return []
        sentences = split_sentences(full_text)
        chunks = build_sentence_chunks(
            sentences,
            max_words=120,
            overlap=2
        )
        if not chunks:
            chunks = [full_text[:1000]]
        return chunks
    except Exception as e:
        print(f"[錯誤] 解析 PDF {paper_id} 失敗: {str(e)}")
        return []


def retrieve_top_k_evidence(paper_id, sentence, top_k=TOP_K):
    try:
        chunks = CHUNK_POOL.get(paper_id, [])
        if not chunks:
            return "查無論文證據內容"
        texts = [c["text"] for c in chunks] # list of chunk texts(str)
        tokenized_text = [t.lower().split() for t in texts]

        bm25 = BM25_POOL.get(paper_id)
        if bm25 is None:
            return "BM25 index missing"

        tokenized_query = sentence.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

        bm25_candidates = [texts[i] for i in top_idx]

        query_emb = EMBED_MODEL.encode(sentence, convert_to_tensor=True)
        cand_embs = EMBED_MODEL.encode(bm25_candidates, convert_to_tensor=True)

        sim = util.cos_sim(query_emb, cand_embs)[0]

        rerank_idx = torch.topk(sim, k=min(top_k, len(bm25_candidates))).indices.tolist()

        final_chunks = [bm25_candidates[i] for i in rerank_idx]

        return "\n...\n".join(final_chunks)

    except Exception as e:
        return f"檢索過程出錯: {str(e)}"
    

# Data Preprocessing and Augmentation
def preprocess_data(df):
    # data loading and preprocessing
    resampled_dfs = []    

    for label_name, group in df.groupby('label_name'):
        current_size = len(group)
        if label_name == 'Number':
            # 目標數量約為 249 * 7 = 1743
            target_size = current_size * 7
            augmented_samples = group.sample(n=(target_size - current_size), replace=True, random_state=42)
            resampled_dfs.append(pd.concat([group, augmented_samples]))
            print(f"[{label_name}] 擴增: {current_size} -> {target_size}")

        elif label_name == 'Temporal':
            # 目標數量約為 130 * 14 = 1820
            target_size = current_size * 14
            augmented_samples = group.sample(n=(target_size - current_size), replace=True, random_state=42)
            resampled_dfs.append(pd.concat([group, augmented_samples]))
            print(f"[{label_name}] 擴增: {current_size} -> {target_size}") 
        else:
            resampled_dfs.append(group)
            print(f"[{label_name}] 保持不變: {current_size}")
    # 打亂資料
    return pd.concat(resampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)


SYSTEM_PROMPT = """你是一個專業的學術論文審查專家，負責檢測論文中的幻覺 (Hallucination) 類型。
我會提供【論文原文證據】與一個【待評估句子】。請仔細比對兩者，判斷該句子屬於以下哪一類幻覺：

- Attribution Failure [0]: 引用錯誤或缺乏出處。
- Entity [1]: 名詞、實體被錯誤替換或插入。
- Number [2]: 數據、年份、維度等數字與原文不符。
- Overgeneralization [3]: 說法過於廣泛，超出證據支持範圍。
- Temporal [4]: 時態、語氣（如 might vs. will）或時間點錯誤。

【規則】：
1. 僅根據提供的證據進行判斷。
2. 輸出格式：只能輸出類別名稱標籤（如 "Number [2]"），不要任何額外文字或解釋。"""


# Formatting function to create ChatML prompts
def formatting_prompts_func(examples):
    # 假設你的 dataset 有 'evidence' (RAG結果), 'text' (評審句), 'label' (答案類別)
    instructions = []
    for evidence, sentence, label, label_name in zip(examples['evidence'], examples['text'], examples['label'], examples['label_name']):
        # 組成內容
        user_content = f"【論文原文證據】：\n{evidence}\n\n【待評估句子】：\n{sentence}\n\n請輸出幻覺類別："
        
        # 轉換成 ChatML 格式串接
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{label_name} [{label}]<|im_end|>"
        )
        instructions.append(text)
    return { "text": instructions }



def main():
    # Prepare training data
    # Get data classes from JSON
    with open("classes.json", "r") as f:
        labels = json.load(f)
        print(labels[:5])  # Print first 5 label mappings

    # Build mapping dictionary
    id_to_label = {item['id']: item['concept'] for item in labels}
    id_to_desc = {item['id']: item['concept_desc'] for item in labels}

    # load training data and map labels to concepts
    train_df = pd.read_csv("train.csv")
    train_df['label_name'] = train_df['label'].map(id_to_label)
    print(f"First 5 rows of training data:\n{train_df.head()}")  # Print first 5 rows of the training data

    # Chunking pdf
    # if chunk pool already exists, load it to save time
    if os.path.exists(CHUNKS_OUTPUT):
        print(f"找到已有的 chunk pool {CHUNKS_OUTPUT}，直接讀取以節省時間...")
        with open(CHUNKS_OUTPUT, "r", encoding="utf-8") as f:
            CHUNK_POOL = json.load(f)
    else:
        print("Processing PDFs to build chunk pool...")
        CHUNK_POOL = {}
        for paper_id in tqdm(train_df['paper_id'].unique(), desc="Processing PDFs"):
            pdf_path = f"paper_evidence/train/{paper_id}.pdf"
            if os.path.exists(pdf_path):
                chunks = process_pdf_to_chunks(pdf_path, paper_id)
                CHUNK_POOL[paper_id] = [
                    {
                        "chunk_id": i,
                        "text": c,
                    }
                    for i, c in enumerate(chunks)
                ]
            else:
                print(f"PDF not found for paper_id {paper_id}, skipping...")
        # 儲存 chunk pool 以供下次訓練使用
        with open(CHUNKS_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(CHUNK_POOL, f, ensure_ascii=False, indent=2)
        print(f"✅ Chunk pool 已建立並儲存至 {CHUNKS_OUTPUT}！")

    for paper_id, chunks in tqdm(CHUNK_POOL.items(), desc="Building BM25 indexes"):
        texts = [c["text"] for c in chunks]
        tokenized_text = [t.lower().split() for t in texts]
        BM25_POOL[paper_id] = BM25Okapi(tokenized_text)

    # retrieve evidence for each training sample
    if os.path.exists(EVIDENCE_OUTPUT):
        print(f"找到已有的 evidence 檔案 {EVIDENCE_OUTPUT}，直接讀取以節省時間...")
        # read pickle file if exists
        train_df = pd.read_pickle(EVIDENCE_OUTPUT)
    else:
        train_df['evidence'] = train_df.apply(
            lambda row: retrieve_top_k_evidence(
                row['paper_id'], 
                row['text'], 
                top_k=TOP_K
                ), 
                axis=1
            )
        print(f"First 5 evidence retrieval results:\n{train_df['evidence'].head()}")
        train_df.to_pickle("./results/train_evidence.pkl")
        train_df.to_csv("./results/train_evidence.csv", index=False, escapechar='\\')
        print("檢索結果已成功儲存！")

    # Handle class imbalance with oversampling
    print("Preprocessing unbalanced training data...")
    processed_df = preprocess_data(train_df)

    # Hugging Face Dataset preparation
    dataset = Dataset.from_pandas(processed_df)
    print(f"[DEBUG] Dataset after preprocessing: {dataset[:3]}")  # Print three example of the dataset
    # Prompt formatting
    train_dataset = dataset.map(formatting_prompts_func, batched=True)

    # 1. Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, 
        max_seq_length=MAX_SEQ_LENGTH, 
        dtype=None, 
        load_in_4bit=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Set up QLoRA parameters
    print("Setting up QLoRA parameters...")
    model = FastLanguageModel.get_peft_model(
        model, 
        r=R, 
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT, 
        bias="none", 
        use_gradient_checkpointing="unsloth",
        random_state = 42,
    )

    # Unsloth 建議的超參數
    training_args = SFTConfig(
        num_train_epochs = 3,
        per_device_train_batch_size = 2, # 配合 Gradient Accumulation
        gradient_accumulation_steps = 4, # 實質 Batch Size = 2 * 4 = 8
        warmup_ratio = 0.1,
        learning_rate = 2e-4, # QLoRA 常用的學習率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit", # 8bit 優化器省記憶體
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = OUTPUT_DIR,
        max_seq_length = MAX_SEQ_LENGTH,
        seed = 42,
    )

    # 使用我們客製化的 Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset, # 你上一階段前處理完的 dataset
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = training_args,
    )

    print("Starting training...")
    trainer.train()

    # 訓練完成後儲存 LoRA 權重
    print(f"Saving model at {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Finished saving model...")

    return




if __name__ == "__main__":
    main()

