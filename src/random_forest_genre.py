
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class GenreClassifier:
    """
    Random Forest classifier for predicting book Genre.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest (default 200).
    test_size : float
        Fraction of data held out for testing (default 0.2).
    random_state : int
        Seed for reproducibility (default 42).
    """

    TARGET = "Genre"
    CATEGORICAL_FEATURES = ["form", "price range", "Reading age"]
    DROP_COLUMNS = ["id_2023", "id_2024", "id_2025", "Book name", "Author",
                    "Rating", "Publishing date"]

    def __init__(self, n_estimators: int = 200, test_size: float = 0.2,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        self.le_target = LabelEncoder()
        self.label_encoders: dict = {}
        self.feature_names: list = []

        # Set after fit
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = None

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "GenreClassifier":
        """Preprocess, split, and train on the supplied DataFrame."""
        X, y = self._preprocess(df)
        self._split_and_train(X, y)
        return self

    def evaluate(self) -> dict:
        """Print accuracy, CV score, and full classification report."""
        self.y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, self.y_pred)
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])
        cv = cross_val_score(self.model, X_full, y_full, cv=5, scoring="accuracy")

        labels_present = np.unique(np.concatenate([self.y_test, self.y_pred]))
        label_names = self.le_target.inverse_transform(labels_present)

        print("=" * 55)
        print("  RANDOM FOREST — GENRE CLASSIFICATION")
        print("=" * 55)
        print(f"  Test Accuracy  : {acc:.4f}")
        print(f"  5-Fold CV Acc  : {cv.mean():.4f} ± {cv.std():.4f}")
        print()
        print(classification_report(self.y_test, self.y_pred,
                                     labels=labels_present,
                                     target_names=label_names,
                                     zero_division=0))
        return {"test_accuracy": acc, "cv_mean": cv.mean(), "cv_std": cv.std()}

    def plot_feature_importance(self, save_path: str = None) -> None:
        """Bar chart of feature importances."""
        importances = self.model.feature_importances_
        feat_df = (pd.DataFrame({"Feature": self.feature_names,
                                  "Importance": importances})
                   .sort_values("Importance", ascending=False))

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=feat_df, x="Importance", y="Feature",
                    hue="Feature", palette="viridis", legend=False, ax=ax)
        ax.set_title("Random Forest — Feature Importances (Genre)")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        plt.show()

    def plot_confusion_matrix(self, top_n: int = 10, save_path: str = None) -> None:
        """Heatmap confusion matrix for the top-N most common genres."""
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)

        top_classes = (pd.Series(self.y_test)
                       .value_counts()
                       .head(top_n)
                       .index
                       .tolist())
        mask = np.isin(self.y_test, top_classes)
        cm = confusion_matrix(self.y_test[mask], self.y_pred[mask],
                               labels=top_classes)
        top_labels = self.le_target.inverse_transform(top_classes)

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=top_labels, yticklabels=top_labels,
                    cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix — Top {top_n} Genres (Random Forest)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        plt.show()

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame):
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Extract derived columns BEFORE dropping originals
        if "Rating" in df.columns:
            df["Rating_num"] = df["Rating"].str.extract(r"([\d.]+)").astype(float)
        if "Publishing date" in df.columns:
            df["pub_year"] = (pd.to_datetime(df["Publishing date"],
                                              dayfirst=True, errors="coerce")
                               .dt.year
                               .fillna(0)
                               .astype(int))

        df = df.drop(columns=[c for c in self.DROP_COLUMNS if c in df.columns])

        df["Reading age"] = df["Reading age"].fillna("unknown")

        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        y = self.le_target.fit_transform(df[self.TARGET])
        X = df.drop(columns=[self.TARGET])
        self.feature_names = list(X.columns)
        return X.values, y

    def _split_and_train(self, X, y):
        # Stratify only when every class has ≥ 2 members
        counts = np.bincount(y)
        stratify = y if (counts >= 2).all() else None
        if stratify is None:
            print("Note: stratification skipped — some genre classes have < 2 samples.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=stratify,
        )
        self.model.fit(self.X_train, self.y_train)
