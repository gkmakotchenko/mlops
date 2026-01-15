import pandas as pd
from sklearn.datasets import fetch_openml
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ds = fetch_openml("titanic", version=1, as_frame=True)
    df = ds.frame.copy()

    df = df.rename(columns={"survived": "target"})
    df["target"] = df["target"].astype(int)

    cols = ["pclass", "sex", "age", "fare", "sibsp", "parch", "embarked", "target"]
    df = df[cols]

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["fare"] = pd.to_numeric(df["fare"], errors="coerce")
    df = df.dropna(subset=["age", "fare", "embarked", "sex", "target"])

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    train = df.iloc[: int(n * 0.7)].copy()
    current = df.iloc[int(n * 0.7):].copy()

    # искусственно создаём дрейф
    rng = np.random.default_rng(42)
    current["age"] = (current["age"] + rng.normal(5, 2, size=len(current))).clip(lower=0)
    current["fare"] = (current["fare"] * 1.15).clip(lower=0)

    train.to_csv(DATA_DIR / "train.csv", index=False)
    current.to_csv(DATA_DIR / "current.csv", index=False)
    print("Saved:", DATA_DIR / "train.csv", "and", DATA_DIR / "current.csv")
    print("Train shape:", train.shape, "Current shape:", current.shape)

if __name__ == "__main__":
    main()
