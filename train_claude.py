import json
import re
import os
from functools import lru_cache

import fitz
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

# ─────────────────────────────────────────────
# Configuration (所有超參數集中管理)
# ─────────────────────────────────────────────
CONFIG = {
    "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "embedder_name": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "max_seq_length": 2048,
    "top_k": 3,
    "chunk_size": 500,
    "overlap": 100,
    "min_chunk_len": 50,
    # QLoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    # Training
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 50,
    "learning_rate": 2e-4,
    "max_steps": 300,
    "output_dir": "outputs",
    "model_save_dir": "hw3_lora_model",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# ─────────────────────────────────────────────
# PDF 快取與語意檢索
# ─────────────────────────────────────────────

# embedder 在此處宣告為 None，延遲到 main() 初始化
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"Loading embedder on {DEVICE}...")
        _embedder = SentenceTransformer(CONFIG["embedder_name"], device=DEVICE)
    return _embedder


@lru_cache(maxsize=256)
def _get_pdf_chunks_and_embeddings(pdf_path: str):
    """
    解析 PDF 並快取 chunks 與 embeddings。
    同一份 PDF 在整個 process 中只解析一次。
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = re.sub(
            r'\s+',
            ' ',
            " ".join(page.get_text("text") for page in doc)
        ).strip()

        if not full_text:
            return [], None

        chunk_size = CONFIG["chunk_size"]
        overlap = CONFIG["overlap"]
        min_len = CONFIG["min_chunk_len"]

        words = full_text.split()
        chunks = [
            " ".join(words[i: i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)
            if len(" ".join(words[i: i + chunk_size]).strip()) > min_len
        ]

        if not chunks:
            return [], None

        embedder = get_embedder()
        embeddings = embedder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        return chunks, embeddings

    except Exception as e:
        print(f"[ERROR] Failed to process {pdf_path}: {e}")
        return [], None


def retrieve_evidence(pdf_path: str, sentence: str, top_k: int = CONFIG["top_k"]) -> str:
    """
    從快取的 PDF chunks 中，以語意相似度取回最相關的 top_k 段落。
    """
    chunks, embeddings = _get_pdf_chunks_and_embeddings(pdf_path)

    if not chunks or embeddings is None:
        return "無法提取 PDF 文字或無相關內容。"

    query_prefix = "Represent this sentence for searching relevant passages: "
    embedder = get_embedder()
    query_embedding = embedder.encode(
        query_prefix + sentence,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]
    top_chunks = [chunks[hit["corpus_id"]] for hit in hits]
    return "\n...\n".join(top_chunks)


# ─────────────────────────────────────────────
# 資料前處理與過採樣
# ─────────────────────────────────────────────

# 各類別的過採樣倍率，未列出的類別保持不變
OVERSAMPLE_MULTIPLIERS = {
    "Number": 3,
    "Temporal": 6,
}


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    針對少數類別進行過採樣，平衡訓練資料分布。
    注意：groupby 必須使用 'label_name' 欄位（字串），
    而非 'label' 欄位（數字 ID）。
    """
    resampled_dfs = []

    for label_name, group in df.groupby("label_name"):  # ← 修正：使用 label_name
        current_size = len(group)
        multiplier = OVERSAMPLE_MULTIPLIERS.get(label_name, 1)

        if multiplier > 1:
            target_size = current_size * multiplier
            augmented = group.sample(
                n=(target_size - current_size),
                replace=True,
                random_state=42
            )
            resampled_dfs.append(pd.concat([group, augmented]))
            print(f"[{label_name}] Oversampled: {current_size} -> {target_size}")
        else:
            resampled_dfs.append(group)
            print(f"[{label_name}] Unchanged: {current_size}")

    return (
        pd.concat(resampled_dfs)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# Prompt 格式化
# ─────────────────────────────────────────────

def formatting_prompts_func(examples: dict, eos_token: str) -> dict:
    """
    將樣本轉換為 ChatML 格式的訓練 prompt。
    label 格式與 SYSTEM_PROMPT 範例對齊：'Number [2]'
    """
    instructions = []
    for evidence, sentence, label, label_name in zip(
        examples["evidence"],
        examples["text"],
        examples["label"],
        examples["label_name"],
    ):
        user_content = (
            f"【論文原文證據】：\n{evidence}\n\n"
            f"【待評估句子】：\n{sentence}\n\n"
            f"請輸出幻覺類別："
        )
        # 格式：label_name [label_id]，與 SYSTEM_PROMPT 一致
        assistant_output = f"{label_name} [{label}]"

        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_output}<|im_end|>{eos_token}"
        )
        instructions.append(text)

    return {"text": instructions}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # 1. 載入標籤對應表
    with open("classes.json", "r") as f:
        labels = json.load(f)
    id_to_label = {item["id"]: item["concept"] for item in labels}
    print(f"Label mapping: {id_to_label}")

    # 2. 載入訓練資料
    train_df = pd.read_csv("train.csv")
    train_df["label_name"] = train_df["label"].map(id_to_label)
    print(f"Training data shape: {train_df.shape}")
    print(train_df.head())

    # 3. RAG 證據提取（PDF 快取，避免重複解析）
    print("Extracting evidence via semantic retrieval...")
    train_df["evidence"] = train_df.apply(
        lambda row: retrieve_evidence(
            f"paper_evidence/{row['paper_id']}.pdf",
            row["text"]
        ),
        axis=1
    )
    print("Evidence extraction done.")
    print(train_df["evidence"].head())

    # 4. 過採樣
    print("Balancing class distribution...")
    processed_df = preprocess_data(train_df)
    print(f"Processed data shape: {processed_df.shape}")

    # 5. 載入模型（延遲到資料準備完成後，節省等待時間）
    print("Loading base model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Applying QLoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 6. 組裝 Dataset 與 Prompt
    dataset = Dataset.from_pandas(processed_df)
    eos_token = tokenizer.eos_token
    train_dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, eos_token),
        batched=True,
    )

    # 7. 訓練
    training_args = TrainingArguments(
        per_device_train_batch_size=CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        learning_rate=CONFIG["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,                    # ← 新增：每 100 步儲存一次 checkpoint
        save_total_limit=2,                # ← 新增：最多保留 2 個 checkpoint
        report_to="none",                  # ← 新增：避免預設嘗試連線 wandb
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_steps=CONFIG["max_steps"],
        output_dir=CONFIG["output_dir"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=2,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # 8. 儲存 LoRA 權重
    model.save_pretrained(CONFIG["model_save_dir"])
    tokenizer.save_pretrained(CONFIG["model_save_dir"])
    print(f"Model saved to '{CONFIG['model_save_dir']}'.")


if __name__ == "__main__":
    main()