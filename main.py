from pathlib import Path
import pandas as pd
from src.mlpipeline import MLPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "raw_data" / "insurance.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "models" / "artifacts.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    pipeline = MLPipeline(
        target_column="charges",
        debug=True,
        artifacts_path=str(ARTIFACTS_PATH)
    )
    results = pipeline.run(df, n_iter_search=5)

    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()
