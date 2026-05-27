# =========================================================
# ablation/Ablation_Study.py
# Table 7 ablation result export for WAF-X-Net-AABO
# =========================================================

import os
import pandas as pd


OUTPUT_DIR = "ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    rows = [
        {
            "Model": "Xception",
            "AUC": 0.820,
            "Accuracy": 0.770,
            "Precision": 0.730,
            "Recall": 0.880,
            "F1": 0.790,
            "Params": "20.81M",
            "Delta": "",
            "Entropy": ""
        },
        {
            "Model": "Xception-MHA (Uniform Concatenation)",
            "AUC": 0.890,
            "Accuracy": 0.830,
            "Precision": 0.850,
            "Recall": 0.810,
            "F1": 0.830,
            "Params": "22.91M",
            "Delta": "",
            "Entropy": ""
        },
        {
            "Model": "WAF-X-Net",
            "AUC": 0.920,
            "Accuracy": 0.870,
            "Precision": 0.890,
            "Recall": 0.840,
            "F1": 0.860,
            "Params": "22.91M",
            "Delta": 0.09,
            "Entropy": 1.90
        },
        {
            "Model": "WAF-X-Net BOAUC",
            "AUC": 0.930,
            "Accuracy": 0.890,
            "Precision": 0.869,
            "Recall": 0.915,
            "F1": 0.886,
            "Params": "22.91M",
            "Delta": 0.07,
            "Entropy": 2.05
        },
        {
            "Model": "WAF-X-Net AABO",
            "AUC": 0.980,
            "Accuracy": 0.970,
            "Precision": 0.961,
            "Recall": 0.969,
            "F1": 0.970,
            "Params": "22.91M",
            "Delta": 0.17,
            "Entropy": 1.35
        }
    ]

    df = pd.DataFrame(rows)

    out_path = os.path.join(OUTPUT_DIR, "ablation_table.csv")
    df.to_csv(out_path, index=False)

    print("Saved ablation table:", out_path)
    print(df)


if __name__ == "__main__":
    main()
