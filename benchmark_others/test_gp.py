from scripts.run_gp import run_gp, run_gp_noise
from scripts.bench_utils import generate_dataset, generate_dataset_test
import pandas as pd
import yaml

df = pd.read_csv("test_set/nc.csv")

X_train, y_train, expr = generate_dataset(df.iloc[100])
X_test, y_test, _ = generate_dataset_test(df.iloc[100])

#print(expr)

with open("benchmark_others/configs/gplearn.yaml") as f:
    cfg = yaml.safe_load(f)

results_no_noise = run_gp(X_train, y_train, X_test, y_test, cfg)
#results_noise = run_gp_noise(X_train, y_train, X_test, y_test, cfg)

print(results_no_noise)
#print(results_noise)