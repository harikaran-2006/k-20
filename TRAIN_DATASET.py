import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import joblib


class MalwareClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_family = LabelEncoder()   # all families (Benign + malware)
        self.le_malware = LabelEncoder()  # malware families only
        self.imputer = SimpleImputer(strategy='mean')

        # Stage 1: Binary classification stacking
        self.stage1 = StackingClassifier(
            estimators=[
                ("LR", LogisticRegression(max_iter=5000, solver="saga")),
                ("RF", RandomForestClassifier(n_estimators=300, max_depth=12,
                                              n_jobs=-1, random_state=42)),
                ("XGB", XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                      subsample=0.8, colsample_bytree=0.8,
                                      eval_metric="logloss", n_jobs=-1, random_state=42))
            ],
            final_estimator=LogisticRegression(),
            n_jobs=-1
        )

        # Stage 2: Malware family stacking
        self.stage2 = StackingClassifier(
            estimators=[
                ("LGBM", LGBMClassifier(n_estimators=500, max_depth=12,
                                        learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.8, random_state=42)),
                ("RF", RandomForestClassifier(n_estimators=400, max_depth=14,
                                              n_jobs=-1, random_state=42)),
                ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128),
                                      max_iter=300, random_state=42))
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            n_jobs=-1
        )

    def preprocess(self, X):
        """Apply imputation + scaling"""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def fit(self, X, y_class, y_family):
        """Train the 2-stage model and evaluate stacking results"""
        # Impute and scale
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Encode labels
        y_bin = (y_class == "Malware").astype(int)
        y_multi = self.le_family.fit_transform(y_family)

        # Fit malware-only label encoder
        malware_families = y_family[y_class == "Malware"]
        self.le_malware.fit(malware_families)

        # Split
        X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
            X_scaled, y_bin, y_multi, test_size=0.2, stratify=y_bin, random_state=42
        )

        # Stage 1: Binary stacking
        self.stage1.fit(X_train, y_train_bin)
        y_pred_bin = self.stage1.predict(X_test)

        print("\n=== Stage 1: Binary Classification Stacking ===")
        print("Accuracy:", accuracy_score(y_test_bin, y_pred_bin))
        print("Precision:", precision_score(y_test_bin, y_pred_bin))
        print("Recall:", recall_score(y_test_bin, y_pred_bin))
        print("F1-score:", f1_score(y_test_bin, y_pred_bin))
        print("\nConfusion Matrix:")
        cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
        sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Malware"],
                    yticklabels=["Benign", "Malware"])
        plt.show()

        # Stage 2: Malware family stacking
        train_mask = (y_train_bin == 1)
        test_mask = (y_test_bin == 1)

        X_train_malware = X_train[train_mask]
        X_test_malware = X_test[test_mask]
        y_train_family = y_train_multi[train_mask]
        y_test_family = y_test_multi[test_mask]

        # Re-encode malware families
        y_train_family_enc = self.le_malware.transform(self.le_family.inverse_transform(y_train_family))
        y_test_family_enc = self.le_malware.transform(self.le_family.inverse_transform(y_test_family))

        if len(np.unique(y_train_family_enc)) > 1:  # guard
            self.stage2.fit(X_train_malware, y_train_family_enc)
            y_pred_family = self.stage2.predict(X_test_malware)

            print("\n=== Stage 2: Malware Family Classification ===")
            print("Accuracy:", accuracy_score(y_test_family_enc, y_pred_family))
            print("Precision:", precision_score(y_test_family_enc, y_pred_family, average='macro'))
            print("Recall:", recall_score(y_test_family_enc, y_pred_family, average='macro'))
            print("F1-score:", f1_score(y_test_family_enc, y_pred_family, average='macro'))

            print("\nConfusion Matrix:")
            cm_stage2 = confusion_matrix(y_test_family_enc, y_pred_family)
            sns.heatmap(cm_stage2, annot=True, fmt="d", cmap="Oranges",
                        xticklabels=self.le_malware.classes_,
                        yticklabels=self.le_malware.classes_)
            plt.show()
        else:
            print("\n[!] Not enough malware family diversity in training set for Stage 2.")

    def predict(self, sample):
        """Run full 2-stage prediction on new sample(s)"""
        sample_processed = self.preprocess(sample)
        stage1_pred = self.stage1.predict(sample_processed)

        results = []
        for i, pred in enumerate(stage1_pred):
            if pred == 0:
                results.append(("Benign", None))
            else:
                fam_pred = self.stage2.predict(sample_processed[i:i+1])[0]
                fam_label = self.le_malware.inverse_transform([fam_pred])[0]
                results.append(("Malware", fam_label))
        return results


# ============================
# Load data & train
# ============================
df = pd.read_csv("/content/0000.csv")
X = df.drop(columns=["Filename", "Class", "Category"])
y_class = df["Class"]
y_family = df["Category"]

# Drop missing targets
valid_indices = df[['Class', 'Category']].dropna().index
X = X.loc[valid_indices]
y_class = y_class.loc[valid_indices]
y_family = y_family.loc[valid_indices]

if __name__ == "__main__":
    model = MalwareClassifier()
    model.fit(X, y_class, y_family)
    joblib.dump(model, "malware_classifier.pkl")
