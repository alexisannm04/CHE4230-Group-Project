
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class PriceClassifier:
    """
    Gradient Boosting classifier for predicting book Price Range.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages (default 200).
    learning_rate : float
        Shrinkage applied to each tree's contribution (default 0.05).
    max_depth : int
        Maximum depth of individual regression estimators (default 4).
    test_size : float
        Fraction of data held out for testing (default 0.2).
    random_state : int
        Seed for reproducibility (default 42).
    """

    TARGET = "price range"
    CATEGORICAL_FEATURES = ["form", "Genre", "Reading age"]
    DROP_COLUMNS = ["id_2023", "id_2024", "id_2025", "Book name", "Author",
                    "Rating", "Publishing date"]

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05,
                 max_depth: int = 4, test_size: float = 0.2,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=4,
            subsample=0.8,
            random_state=random_state,
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

    def fit(self, df: pd.DataFrame) -> "PriceClassifier":
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
        print("  GRADIENT BOOSTING — PRICE RANGE CLASSIFICATION")
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
                    hue="Feature", palette="magma", legend=False, ax=ax)
        ax.set_title("Gradient Boosting — Feature Importances (Price Range)")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        plt.show()

    def plot_confusion_matrix(self, save_path: str = None) -> None:
        """Heatmap confusion matrix across all price buckets."""
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)

        labels_present = np.unique(np.concatenate([self.y_test, self.y_pred]))
        label_names = self.le_target.inverse_transform(labels_present)
        cm = confusion_matrix(self.y_test, self.y_pred, labels=labels_present)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=label_names, yticklabels=label_names,
                    cmap="Oranges", ax=ax)
        ax.set_title("Confusion Matrix — Price Range (Gradient Boosting)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        plt.show()

    def plot_learning_curve(self, save_path: str = None) -> None:
        """Stage-wise training deviance — shows if the model is converging."""
        train_scores = self.model.train_score_

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_scores, color="darkorange", label="Training deviance")
        ax.set_xlabel("Boosting Iterations")
        ax.set_ylabel("Deviance")
        ax.set_title("Gradient Boosting — Training Deviance over Iterations")
        ax.legend()
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
        counts = np.bincount(y)
        stratify = y if (counts >= 2).all() else None
        if stratify is None:
            print("Note: stratification skipped — some price classes have < 2 samples.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=stratify,
        )
        self.model.fit(self.X_train, self.y_train)
