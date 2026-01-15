import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for numeric feature."""
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)

    # одинаковые границы по expected
    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.quantile(expected, quantiles)
    breaks[0] = -np.inf
    breaks[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=breaks)
    act_counts, _ = np.histogram(actual, bins=breaks)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    # защита от нулей
    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    current = pd.read_csv(DATA_DIR / "current.csv")

    features = ["age", "fare"]  # минимально и достаточно для демонстрации
    scores = {f: psi(train[f], current[f]) for f in features}
    avg_psi = float(np.mean(list(scores.values())))

    threshold = float(Path("data/drift_threshold.txt").read_text().strip()) if Path("data/drift_threshold.txt").exists() else 0.2
    drift = avg_psi > threshold

    print("PSI per feature:", scores)
    print("AVG_PSI:", avg_psi, "THRESHOLD:", threshold, "DRIFT:", drift)

    # флаг для Airflow
    Path("data/drift_flag.txt").write_text("1" if drift else "0")

if __name__ == "__main__":
    main()
