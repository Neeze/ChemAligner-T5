import csv
from src.metric_evaluator.translation_metrics import Mol2Text_translation

def evaluate_from_csv(csv_path):
    evaluator = Mol2Text_translation()

    gt_selfies = []
    pred_selfies = []

    # Đọc dữ liệu từ file CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_selfies.append(row['gt_selfie'])
            pred_selfies.append(row['pred_selfie'])

    # Tính metrics
    metrics_result = evaluator(pred_selfies, gt_selfies)

    # In kết quả
    print("=== Evaluation Results ===")
    for k, v in metrics_result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    csv_path = "results/BioT5p_MaMMUT.csv"  # đường dẫn đến file CSV kết quả
    evaluate_from_csv(csv_path)
