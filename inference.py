import json
import pandas as pd
import os
import re
import torch
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from unsloth import FastLanguageModel  

# ==========================================
# 0. 基本設定與模型載入
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "./checkpoint"

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME, 
    max_seq_length = MAX_SEQ_LENGTH, 
    dtype = None, 
    load_in_4bit = True
)

FastLanguageModel.for_inference(model)

# ==========================================
# 1. RAG 檢索模組 (確保與訓練時 100% 一致)
# ==========================================
print("載入 Snowflake Embedding 模型...")
embedder = SentenceTransformer('Snowflake/snowflake-arctic-embed-l-v2.0', device=DEVICE)

PDF_CACHE = {}

def optimized_extract_evidence(pdf_path, sentence, top_k=3, chunk_size=500, overlap=100):
    try:
        if not os.path.exists(pdf_path): return "PDF 檔案不存在"
        
        if pdf_path not in PDF_CACHE:
            doc = fitz.open(pdf_path)
            full_text = " ".join([page.get_text("text") for page in doc])
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            if not full_text: return "無法提取 PDF 文字"

            words = full_text.split()
            chunks = [ " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - overlap) if len(" ".join(words[i : i + chunk_size]).strip()) > 50 ]
            if not chunks: return full_text[:2000]

            chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True).cpu()
            PDF_CACHE[pdf_path] = {'chunks': chunks, 'embeddings': chunk_embeddings}

        chunks = PDF_CACHE[pdf_path]['chunks']
        chunk_embeddings = PDF_CACHE[pdf_path]['embeddings'].to(DEVICE)

        query_prefix = "Represent this sentence for searching relevant passages: "
        query_embedding = embedder.encode(query_prefix + sentence, convert_to_tensor=True)

        hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)[0]
        top_chunks = [chunks[hit['corpus_id']] for hit in hits]
        return "\n...\n".join(top_chunks)

    except Exception as e:
        return f"PDF 解析或檢索錯誤: {str(e)}"

# ==========================================
# 2. 推論與解析邏輯
# ==========================================
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

def predict_label(evidence, sentence):
    # 組合 Prompt
    user_content = f"【論文原文證據】：\n{evidence}\n\n【待評估句子】：\n{sentence}\n\n請輸出幻覺類別："
    
    # 轉換為模型的輸入格式 (注意結尾加上 <|im_start|>assistant\n 來引導模型作答)
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)
    
    # 產出回答 (設定 max_new_tokens=10 即可，因為我們只要短短的標籤)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10, 
        use_cache=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解碼模型輸出的字串
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 切割出 assistant 回答的部分
    response = generated_text.split("請輸出幻覺類別：")[-1].strip()
    
    # 🛠️ 關鍵：使用 Regex 抓取中括號內的數字
    match = re.search(r'\[(\d+)\]', response)
    if match:
        return int(match.group(1))
    else:
        # 如果模型崩潰沒輸出數字，預設給出佔比最大的類別 (例如 0) 以降低損失
        print(f"解析失敗，原始輸出：{response}。預設填入 0。")
        return 0

# ==========================================
# 3. 主程式：產出 Kaggle CSV
# ==========================================
def main():
    # 假設測試集檔名為 test.csv (請依據作業實際檔名修改)
    # 測試集應該會有 'id', 'paper_id', 'text' 這些欄位
    TEST_FILE = "test.csv" 
    STUDENT_ID = "你的學號"  # 記得填寫你的學號！
    OUTPUT_FILE = f"hw3_{STUDENT_ID}.csv"
    
    print(f"開始讀取測試資料 {TEST_FILE} ...")
    test_df = pd.read_csv(TEST_FILE)
    
    predictions = []
    
    # 逐筆進行推論
    for index, row in test_df.iterrows():
        print(f"正在處理第 {index + 1}/{len(test_df)} 筆資料...")
        
        # 1. 執行 RAG 抽取
        pdf_path = f"paper_evidence/{row['paper_id']}.pdf"
        evidence = optimized_extract_evidence(pdf_path, row['text'])
        
        # 2. 模型推論並解析 ID
        pred_id = predict_label(evidence, row['text'])
        predictions.append(pred_id)
        
    # 儲存結果
    test_df['label'] = predictions
    
    # Kaggle 通常只要求 id 和 label 兩個欄位
    submission_df = test_df[['id', 'label']]
    submission_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ 推論完成！結果已儲存至 {OUTPUT_FILE}，祝你拿高分！")

if __name__ == "__main__":
    main()