import pandas as pd
from sklearn.metrics import f1_score, classification_report

TEST_FILE = "dev.csv"
STUDENT_ID = "dev_result"  # 記得填寫你的學號！
OUTPUT_FILE = f"hw3_{STUDENT_ID}.csv"


print(f"開始讀取結果資料 {OUTPUT_FILE} ...")
submission_df = pd.read_csv(OUTPUT_FILE)
test_df = pd.read_csv(TEST_FILE)

# Merge submission with test_df to get paper_id and text
merged_df = pd.merge(submission_df, test_df, on='id', how='left')
# Print first 5 rows to verify
print(f"First 5 rows of merged data:\n{merged_df.head()}")

'''
# calculate Macro f1-score
if 'label_x' in merged_df.columns and 'label_y' in merged_df.columns:
    y_pred = merged_df['label_x']
    y_true = merged_df['label_y']
else:
    # 如果合併時沒有改名（可能其中一方的欄位名稱不叫 label），請根據實際情況修改這裡
    print("找不到 label_x 或 label_y，請確認預測與真實標籤的欄位名稱！")
    print("目前擁有的欄位：", merged_df.columns.tolist())
    # y_pred = merged_df['你的預測欄位']
    # y_true = merged_df['你的真實欄位']

# 1. 計算總體 Macro F1 分數
macro_f1 = f1_score(y_true, y_pred, average='macro')
print("\n" + "="*40)
print(f"🏆 最終 Macro F1-score: {macro_f1:.4f}")
print("="*40 + "\n")

# 2. 印出詳細的分類報告 (強烈建議！)
# 這可以讓你清楚看到模型在哪個幻覺類別 (0~4) 表現最差，方便對症下藥
print("📊 各類別詳細表現報告：")
target_names = [
    "Attribution Failure (0)", 
    "Entity (1)", 
    "Number (2)", 
    "Overgeneralization (3)", 
    "Temporal (4)"
]

# 避免有類別剛好完全沒預測到而報錯，這裡設定 zero_division=0
report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
print(report)
'''