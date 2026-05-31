# =============================================
# FILE 1: train_model.py
# PURPOSE: Extract features + Train ML Model
# TOPICS: NumPy, OpenCV, ML Classification,
#         OOP, File Handling, Modules
# =============================================

import cv2
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---- FUNCTION: Load Images from Folder ----
def load_images(folder, label):
    data = []
    labels = []
    
    print(f"Loading images from: {folder}")
    
    for filename in os.listdir(folder):        # File Handling
        img_path = os.path.join(folder, filename)
        try:                                   # Exception Handling
            img = cv2.imread(img_path)         # OpenCV read image
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))    # OpenCV resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            arr = np.array(img).flatten()      # NumPy array + flatten
            data.append(arr)
            labels.append(label)
        except Exception as e:
            print(f"Skipping {filename}: {e}") # Exception Handling

    print(f"Loaded {len(data)} images from {folder}")
    return data, labels

# ---- STEP 1: Load Dataset ----
mask_data,   mask_labels   = load_images("dataset/with_mask",    1)
nomask_data, nomask_labels = load_images("dataset/without_mask", 0)

# ---- STEP 2: Prepare NumPy Arrays ----
X = np.array(mask_data + nomask_data)
y = np.array(mask_labels + nomask_labels)
print(f"\nTotal samples: {len(X)}")

# ---- STEP 3: Split into Train and Test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing  samples: {len(X_test)}")

# ---- STEP 4: Train 3 ML Models ----
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000),
    "Decision Tree"       : DecisionTreeClassifier(),
    "Random Forest"       : RandomForestClassifier(n_estimators=100)
}

accuracies = {}
trained_models = {}

print("\n--- Training Models ---")
for name, model in models.items():
    model.fit(X_train, y_train)              # Train model
    preds = model.predict(X_test)            # Predict
    acc   = accuracy_score(y_test, preds) * 100
    accuracies[name] = acc
    trained_models[name] = model
    print(f"{name}: {acc:.2f}% accuracy")

# ---- STEP 5: Save Best Model ----
best_name  = max(accuracies, key=accuracies.get)
best_model = trained_models[best_name]

os.makedirs("model", exist_ok=True)
with open("model/mask_detector_model.pkl", "wb") as f:
    pickle.dump(best_model, f)               # File Handling

print(f"\nBest Model Saved: {best_name} ({accuracies[best_name]:.2f}%)")

# ---- STEP 6: Plot Accuracy Graph ----
plt.figure(figsize=(8, 5))
bars = plt.bar(
    accuracies.keys(),
    accuracies.values(),
    color=["#2196F3", "#9C27B0", "#4CAF50"]
)
for bar, acc in zip(bars, accuracies.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() - 3,
             f"{acc:.1f}%", ha="center",
             color="white", fontweight="bold")

plt.title("ML Model Accuracy Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("model_accuracy.png")
plt.show()
print("Chart saved as model_accuracy.png")