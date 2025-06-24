from scripts.run_sklearngp import run_sklearngp
from scripts.bench_utils import generate_dataset
import pandas as pd
import yaml

df = pd.read_csv("test_set/nc.csv")

X, y, expr = generate_dataset(df.iloc[100])

print(expr)

with open("benchmark_others/configs/sklearngp.yaml") as f:
    cfg = yaml.safe_load(f)

results = run_sklearngp(X, y, cfg)
print(results)