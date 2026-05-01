import json
import pandas as pd
import os
import re
import torch
from tqdm import tqdm
from datasets import Dataset
import torch.nn as nn
import fitz  # PyMuPDF
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
CHUNK_SIZE = 300
OVERLAP = 60
CHUNK_POOL = {}
# make directory for results if not exists
if not os.path.exists("./results"):
    os.makedirs("./results")
CHUNKS_OUTPUT = "./results/chunks.json"
EVIDENCE_OUTPUT = "./results/train_evidence.pkl"

# Load embedding model for optimized evidence retrieval
embedder = SentenceTransformer(
    'Snowflake/snowflake-arctic-embed-l-v2.0', 
    device=DEVICE
)

def process_pdf_to_chunks(pdf_path, paper_id, chunk_size=CHUNK_SIZE, overlap=OVERLAP):  
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            try:
                pages_text.append(page.get_text("text"))
            except Exception as e:
                continue
        full_text = " ".join(pages_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        if not full_text:
            return []
        words = full_text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 10:
                chunks.append(chunk)
        if not chunks:
            chunks = [full_text[:2000]]
        return chunks

    except Exception as e:
        print(f"[錯誤] 解析 PDF {paper_id} 失敗: {str(e)}")
        return []


def build_chunk_pool(train_df):
    chunk_pool = {}
    unique_paper_ids = train_df['paper_id'].unique()
    print(f"開始處理，共有 {len(unique_paper_ids)} 篇獨立 PDF 需要切塊...")

    for paper_id in tqdm(unique_paper_ids, desc="PDF Chunking"):
        pdf_path = f"paper_evidence/train/{paper_id}.pdf"
        
        # 直接處理，不需要再檢查是否在 chunk_pool 裡面了，因為我們已經去除了重複項
        chunk_pool[paper_id] = process_pdf_to_chunks(pdf_path, paper_id)
        
    print("\n✅ 所有 PDF 已處理完畢，Chunk Pool 建立完成！")
    return chunk_pool


def retrieve_top_k_evidence(chunk_embeding, chunk_list, sentence, top_k=TOP_K):
    try:
        # 準備檢索
        current_embeddings = chunk_embeding.to(DEVICE)
        query_prefix = "Represent this sentence for searching relevant passages: "
        query_embedding = embedder.encode(
            query_prefix + sentence, 
            convert_to_tensor=True, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )

        # 執行語意比對
        hits = util.semantic_search(query_embedding, current_embeddings, top_k=top_k)[0]
        
        # 組合結果
        top_chunks = [chunk_list[hit['corpus_id']] for hit in hits]
        return "\n...\n".join(top_chunks)

    except Exception as e:
        return f"檢索過程出錯: {str(e)}"

'''
# Optimized evidence extraction and retrieval function
def optimized_extract_evidence(pdf_path, sentence, top_k=TOP_K, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    try:
        # print(f"[DEBUG] 正在處理 PDF: {pdf_path}，查詢句子: {sentence[:50]}...")
        if pdf_path not in PDF_CACHE:
            if not os.path.exists(pdf_path):
                return f"PDF not found: {pdf_path}"

            doc = fitz.open(pdf_path)
            # print(f"[DEBUG] 已從 Cache 載入 PDF: {pdf_path}")
            full_text = " ".join([page.get_text("text") for page in doc])
            
            # 移除多餘的換行與空白，將文本壓平，避免斷句問題
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            if not full_text:
                return "無法提取 PDF 文字"

            # 2. 滑動窗口切塊 (Sliding Window Chunking by Words)
            # 這裡以「單字(Words)」為單位進行切塊，比單純用字元切塊更具備語意完整性
            words = full_text.split()
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])
                if len(chunk.strip()) > 50: # 過濾掉太短的雜訊塊
                    chunks.append(chunk)

            if not chunks:
                return full_text[:2000]

            # 3. 語意檢索 (Semantic Retrieval)
            # 將切塊和查詢句子轉換為 Dense Embeddings
            chunk_embeddings = embedder.encode(
                chunks, 
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
                )
            PDF_CACHE[pdf_path] = {
                'chunks': chunks,
                'embeddings': chunk_embeddings.cpu()  # 先放在 CPU，等需要時再移到 GPU
            }

        print(f"[DEBUG] 已載入 PDF 並切塊: {pdf_path}，共 {len(PDF_CACHE[pdf_path]['chunks'])} 個塊")
        chunks = PDF_CACHE[pdf_path]['chunks']
        chunk_embeddings = PDF_CACHE[pdf_path]['embeddings'].to(DEVICE)
        query_prefix = "Represent this sentence for searching relevant passages: "
        query_embedding = embedder.encode(query_prefix + sentence, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

        # 使用 Cosine Similarity 找出最相似的 Top-K
        # util.semantic_search 已經高度優化過，速度遠比 sklearn 快
        hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)[0]

        # 4. 組合最終證據
        top_chunks = [chunks[hit['corpus_id']] for hit in hits]
        
        # 透過控制 top_k 和 chunk_size 來自然限制長度，取代暴力的 string[:4000]
        relevant_evidence = "\n...\n".join(top_chunks)
        
        return relevant_evidence

    except Exception as e:
        return f"PDF 解析或檢索錯誤: {str(e)}"
'''

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

    # 3. Load embedding model for optimized evidence retrieval
    embedder = SentenceTransformer(
        'Snowflake/snowflake-arctic-embed-l-v2.0', 
        device=DEVICE
    )

    # Chunking pdf
    # if chunk pool already exists, load it to save time
    if os.path.exists(CHUNKS_OUTPUT):
        print(f"找到已有的 chunk pool {CHUNKS_OUTPUT}，直接讀取以節省時間...")
        with open(CHUNKS_OUTPUT, "r", encoding="utf-8") as f:
            CHUNK_POOL = json.load(f)
    else:
        print("Processing PDFs to build chunk pool...")
        CHUNK_POOL = build_chunk_pool(train_df)
        # 儲存 chunk pool 以供下次訓練使用
        with open(CHUNKS_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(CHUNK_POOL, f, ensure_ascii=False)
        print(f"✅ Chunk pool 已建立並儲存至 {CHUNKS_OUTPUT}！")


    # retrieve evidence for each training sample
    if os.path.exists(EVIDENCE_OUTPUT):
        print(f"找到已有的 evidence 檔案 {EVIDENCE_OUTPUT}，直接讀取以節省時間...")
        # read pickle file if exists
        train_df = pd.read_pickle(EVIDENCE_OUTPUT)
    else:
        all_chunks = sum(CHUNK_POOL.values(), [])  # 將所有 chunks 合併成一個大列表
        print(f"總共從 PDF 中提取了 {len(all_chunks)} 個 chunks")
        print("正在為所有 chunks 計算 embeddings，這可能需要一些時間...")
        all_chunks_embeddings = embedder.encode(
            all_chunks, 
            convert_to_tensor=True, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )

        
        # retrieve evidence for each training sample and store in a new column
        train_df['evidence'] = train_df.apply(
            lambda row: retrieve_top_k_evidence(
                all_chunks_embeddings, 
                all_chunks, row['text'], 
                top_k=TOP_K
                ), 
                axis=1
            )
        print(f"First 5 evidence retrieval results:\n{train_df['evidence'].head()}")
        train_df.to_pickle("./results/train_evidence.pkl")
        train_df.to_csv("./results/train_evidence.csv", index=False, escapechar='\\')
        print("檢索結果已成功儲存！")
        del all_chunks_embeddings  # 用完就丟，釋放 GPU 記憶體


    ''' 
    # RAG evidence extraction and retrieval
    EVIDENCE_CACHE_FILE = "train_with_evidence.pkl"
    if os.path.exists(EVIDENCE_CACHE_FILE):
        print(f"找到已有的 RAG 結果 {EVIDENCE_CACHE_FILE}，直接讀取以節省時間...")
        train_df = pd.read_pickle(EVIDENCE_CACHE_FILE)
    else:
        print("Extracting and retrieving evidence from PDFs (有了 Cache 機制，現在會快很多!)...")
        train_df['evidence'] = train_df.apply(lambda row: optimized_extract_evidence(f"paper_evidence/train/{row['paper_id']}.pdf", row['text']), axis=1)
        # 算完馬上存檔！
        train_df.to_pickle(EVIDENCE_CACHE_FILE)
        print("✅ Evidence 擷取完成，已儲存至 train_with_evidence.pkl！")    
    print(f"First 5 evidence retrieval results:\n{train_df['evidence'].head()}")
    '''
    # cleanup to free GPU memory before training
    del embedder
    torch.cuda.empty_cache()
    import gc
    gc.collect()

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
        output_dir = "outputs",
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

