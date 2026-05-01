import pandas as pd
from sklearn.metrics import f1_score, classification_report

PRED_PATH = "./results/dev_pred.csv"
LABEL_PATH = "dev.csv"

def evaluate_macro_f1():
    try:
        # 1. 讀取真實標籤與預測結果
        print(f"正在載入資料...\n真實標籤: {LABEL_PATH}\n預測結果: {PRED_PATH}")
        dev_df = pd.read_csv(LABEL_PATH)
        pred_df = pd.read_csv(PRED_PATH)

        # 防呆檢查：確保預測筆數與真實筆數完全對應
        if len(dev_df) != len(pred_df):
            print(f"[錯誤] 資料筆數不一致！真實標籤有 {len(dev_df)} 筆，預測結果有 {len(pred_df)} 筆。")
            return

        # 擷取真實與預測的 label 陣列
        y_true = dev_df['label'].tolist()
        y_pred = pred_df['label'].tolist() 

        # 2. 計算核心指標：Macro F1 Score
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        print("="*50)
        print(f"🌟 最終 Validation Macro F1 Score: {macro_f1:.4f}")
        print("="*50)

        # 3. 助教加碼：印出各類別的詳細報告，抓出拖後腿的類別！
        print("\n📊 各類別詳細評估報告 (Classification Report):")
        
        # 根據 HW3 的 0-4 類別定義
        target_names = [
            'Attribution Failure [0]', 
            'Entity [1]', 
            'Number [2]', 
            'Overgeneralization [3]', 
            'Temporal [4]'
        ]
        
        # digits=4 可以顯示到小數點後四位，看分更精準
        report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        print(report)

    except FileNotFoundError as e:
        print(f"\n[錯誤] 找不到檔案：{e.filename}。請確認路徑是否正確！")
    except KeyError as e:
        print(f"\n[錯誤] 欄位缺失：找不到 {e} 欄位。請確保你的 PRED_PATH CSV 中包含 'label' 欄位。")
    except Exception as e:
        print(f"\n[錯誤] 評估過程中發生未知錯誤: {e}")

if __name__ == "__main__":
    evaluate_macro_f1()