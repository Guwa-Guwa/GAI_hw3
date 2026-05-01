import os
import json
import fitz
import pandas as pd
import torch
from unsloth import FastLanguageModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
ADAPTER_DIR = "./adapter_checkpoint" 
MAX_SEQ_LENGTH = 2048
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
    print("載入基礎模型與 LoRA 權重 (GPU Mode)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model.load_adapter(ADAPTER_DIR)
    FastLanguageModel.for_inference(model)

    print("讀取 Test 資料與類別對應表...")
    test_df = pd.read_csv("test.csv")
    
    with open('classes.json', 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
    
    label_to_id = {item['concept']: int(item['id']) for item in classes_info}

    # 與 train.py 完全一致的 Prompt 結構 (去除 {label})
    prompt_template = """<|im_start|>system
你是一個嚴謹的學術論文評審輔助系統。請仔細比對以下「評審句子」與「論文段落」，判斷該評審句子屬於哪一種幻覺類型。
可選的幻覺類型包含：Attribution Failure, Entity, Number, Overgeneralization, Temporal。
請直接輸出幻覺類型的英文名稱，不要輸出其他多餘文字。
<|im_end|>
<|im_start|>user
論文段落：
{evidence}

評審句子：
{sentence}
<|im_end|>
<|im_start|>assistant
### Answer:
"""
    
    predictions = []
    print("開始進行測試集推論...")
    for index, row in test_df.iterrows():
        pdf_path = f"paper_evidence/{row['paper_id']}.pdf" 
        evidence = extract_and_retrieve_evidence(pdf_path, row['text'])
        
        prompt = prompt_template.format(evidence=evidence, sentence=row['text'])
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id, max_length=None)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        pred_id = label_to_id.get(generated_text, 0)
        if generated_text not in label_to_id:
            for label, label_id in label_to_id.items():
                if label.lower() in generated_text.lower():
                    pred_id = label_id
                    break
                    
        predictions.append(pred_id)
        
        if index > 0 and index % 100 == 0:
            print(f"已預測 {index} 筆資料...")

    print("推論完成，正在生成 Kaggle 繳交檔...")
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    # 輸出符合 Kaggle 與 E3 規定的檔案名稱
    output_filename = "hw3_314512094.csv"
    submission_df.to_csv(output_filename, index=False)
    
    print(f"成功儲存檔案：{output_filename}！現在可以上傳 Kaggle 拿高分了！")

if __name__ == "__main__":
    main()