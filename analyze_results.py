# =============================================
# FILE 3: analyze_results.py
# PURPOSE: Analyze detection log with graphs
# TOPICS: Pandas, Matplotlib, Seaborn
# =============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- STEP 1: Load CSV using Pandas ----
if not os.path.exists("detection_log.csv"):
    print("Run detect_mask_webcam.py first to generate log!")
    exit()

df = pd.read_csv("detection_log.csv")
print("=== Detection Log ===")
print(df.head(10))                             # Show first 10 rows
print("\n=== Summary ===")
print(df["Status"].value_counts())             # Pandas value_counts

# ---- STEP 2: Bar Chart using Seaborn ----
plt.figure(figsize=(7, 5))
sns.countplot(
    data=df, x="Status",
    palette={"Mask": "#4CAF50", "No Mask": "#F44336"}
)
plt.title("Mask vs No Mask Detection Count", fontsize=14)
plt.xlabel("Detection Status")
plt.ylabel("Number of Detections")
plt.tight_layout()
plt.savefig("detection_count.png")
plt.show()

# ---- STEP 3: Pie Chart using Matplotlib ----
counts = df["Status"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90,
    explode=(0.05, 0.05)
)
plt.title("Detection Distribution", fontsize=14)
plt.tight_layout()
plt.savefig("detection_pie.png")
plt.show()

print("\nAnalysis complete!")
print("Charts saved: detection_count.png, detection_pie.png")