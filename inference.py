import json
import pandas as pd
import os
import re
import nltk
import torch
from tqdm import tqdm
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from unsloth import FastLanguageModel  
from sentence_transformers import SentenceTransformer, util

# 確保 NLTK 元件存在
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# 0. 基本設定與模型載入
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 2048
MODEL_PATH = "./checkpoint"
TOP_K = 3

TEST_CSV_PATH = "dev.csv"
TEST_PDF_DIR = "paper_evidence/dev"
SUBMISSION_FILE = "./results/dev_pred.csv" 

# RAG 暫存池
CHUNK_POOL = {}
BM25_POOL = {}

# 載入 Embedding 模型
print("Loading Embedding Model...")
EMBED_MODEL = SentenceTransformer('Snowflake/snowflake-arctic-embed-l-v2.0', device=DEVICE)


def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def build_sentence_chunks(sentences, max_words=120, overlap_words=15):
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    for sent in sentences:
        words = sent.split()
        sent_word_count = len(words)
        if current_word_count + sent_word_count > max_words and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk_sentences):
                s_words = len(s.split())
                if overlap_count + s_words <= overlap_words:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_words
                else:
                    if not overlap_sentences:
                        overlap_sentences.insert(0, s)
                    break
            current_chunk_sentences = overlap_sentences
            current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
        current_chunk_sentences.append(sent)
        current_word_count += sent_word_count
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    return chunks

def process_pdf_to_chunks(pdf_path, paper_id):
    try:
        doc = fitz.open(pdf_path)
        pages_text = [page.get_text("text") for page in doc]
        full_text = "\n".join(pages_text) 
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        if not full_text: return []
        sentences = split_sentences(full_text)
        chunks = build_sentence_chunks(sentences, max_words=120, overlap_words=15)
        return chunks if chunks else [full_text[:1000]]
    except Exception as e:
        print(f"[警告] 解析 PDF {paper_id} 失敗: {str(e)}")
        return []

def clean_tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text).lower()
    return text.split()

def retrieve_top_k_evidence(paper_id, sentence, top_k=TOP_K):
    clean_paper_id = str(paper_id).strip()
    try:
        chunks = CHUNK_POOL.get(clean_paper_id, [])
        if not chunks:
            return "（無法解析此論文的原文證據，請直接根據待評估句子的語意合理性進行判斷。）"
            
        texts = [c["text"] for c in chunks]
        bm25 = BM25_POOL.get(clean_paper_id)
        if bm25 is None: return "BM25 index missing"

        tokenized_query = clean_tokenize(sentence)
        scores = bm25.get_scores(tokenized_query)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        bm25_candidates = [texts[i] for i in top_idx]

        query_emb = EMBED_MODEL.encode(sentence, convert_to_tensor=True, show_progress_bar=False)
        cand_embs = EMBED_MODEL.encode(bm25_candidates, convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(query_emb, cand_embs)[0]
        rerank_idx = torch.topk(sim, k=min(top_k, len(bm25_candidates))).indices.tolist()
        final_chunks = [bm25_candidates[i] for i in rerank_idx]
        return "\n...\n".join(final_chunks)
    except Exception as e:
        return f"檢索過程出錯: {str(e)}"


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

def main():
    # 讀取測試集
    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    # 1. 建立測試集的 Chunk 與 BM25 Pool
    print("Processing Test PDFs...")
    for paper_id in tqdm(test_df['paper_id'].unique(), desc="Parsing Test PDFs"):
        str_paper_id = str(paper_id).strip()
        pdf_path = f"{TEST_PDF_DIR}/{str_paper_id}.pdf"
        
        if os.path.exists(pdf_path):
            chunks = process_pdf_to_chunks(pdf_path, str_paper_id)
            CHUNK_POOL[str_paper_id] = [{"chunk_id": i, "text": c} for i, c in enumerate(chunks)]
        else:
            print(f"PDF not found for test paper_id {str_paper_id}")

    for paper_id, chunks in tqdm(CHUNK_POOL.items(), desc="Building BM25 indexes for Test"):
        texts = [c["text"] for c in chunks]
        tokenized_texts = [clean_tokenize(t) for t in texts]
        BM25_POOL[paper_id] = BM25Okapi(tokenized_texts)

    # 2. 檢索 Test Set 證據
    print("Retrieving evidence for test set...")
    tqdm.pandas(desc="Retrieving Evidence")
    test_df['evidence'] = test_df.progress_apply(
        lambda row: retrieve_top_k_evidence(row['paper_id'], row['text'], top_k=TOP_K), axis=1
    )

    # 用完 Embedding 模型後清空 VRAM，讓給 LLM
    global EMBED_MODEL
    del EMBED_MODEL
    torch.cuda.empty_cache()

    # 3. 載入微調後的模型
    print(f"Loading Fine-tuned model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH, # 這裡直接讀取你存檔的目錄
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    # 開啟 Unsloth 專屬的推論加速模式！(2倍速)
    FastLanguageModel.for_inference(model)

    predictions = []
    
    print("Starting generation...")
    # 4. 開始逐筆生成
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        user_content = f"【論文原文證據】：\n{row['evidence']}\n\n【待評估句子】：\n{row['text']}\n\n請輸出幻覺類別："
        
        # 組裝 Prompt
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)
        
        # 讓模型生成答案 (設定 max_new_tokens=15 即可，因為只需輸出類別)
        outputs = model.generate(**inputs, max_new_tokens=15, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        
        # 解碼輸出
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # 取出 assistant 的回答部分
        answer = generated_text.split("assistant\n")[-1].strip()
        
        # 使用正則表達式萃取方括號內的數字 ID (例如從 "Number [2]" 抓出 2)
        match = re.search(r'\[(\d+)\]', answer)
        if match:
            pred_label = int(match.group(1))
        else:
            # 如果模型胡言亂語沒照格式，找字串裡的類別名稱對應到 ID
            if "Attribution Failure" in answer:
                pred_label = 0
            elif "Entity" in answer:
                pred_label = 1
            elif "Number" in answer:
                pred_label = 2
            elif "Overgeneralization" in answer:
                pred_label = 3
            elif "Temporal" in answer:
                pred_label = 4
            else:
                pred_label = 0 
            print(f"\n[警告] 模型輸出格式異常: {answer} -> 對應為 {pred_label}")
            
        predictions.append(pred_label)

    # 5. 輸出提交檔
    test_df['id'] = test_df.index # 如果有 id 欄位請保留    
    test_df['label'] = predictions
    submission_df = test_df[['label']] # 請依據 HW3 規定調整欄位，通常至少需要 ID 與 label
    
    # 👉 記得把 student_id 換成你的學號！
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"✅ 推論完成！已產生 Kaggle 提交檔：{SUBMISSION_FILE}")

if __name__ == "__main__":
    main()