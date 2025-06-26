from scripts.run_dsr import run_dsr, run_dsr_noise
from scripts.bench_utils import generate_dataset
import pandas as pd
import json


df = pd.read_csv("test_set/nc.csv")

X_train, y_train, expr = generate_dataset(df.iloc[100])
X_test, y_test, _ = generate_dataset(df.iloc[100])


with open("benchmark_others/configs/dsr.json") as f:
    cfg = json.load(f)


results_no_noise = run_dsr(X_train, y_train, X_test, y_test, cfg)
results_noise = run_dsr_noise(X_train, y_train, X_test, y_test, cfg)

print(results_no_noise)
print(results_noise)
