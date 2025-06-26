from scripts.run_sklearngp import run_sklearngp,run_sklearngp_noise
from scripts.bench_utils import generate_dataset, generate_dataset_test
import pandas as pd
import yaml

df = pd.read_csv("/home/scur1229/nesymres/test_set/nc.csv")

X_train, y_train, expr = generate_dataset(df.iloc[102])
X_test, y_test, _ = generate_dataset_test(df.iloc[102])

#print(expr)

with open("/home/scur1229/nesymres/benchmark_others/configs/sklearngp.yaml") as f:
    cfg = yaml.safe_load(f)

results_no_noise = run_sklearngp(X_train, y_train, X_test, y_test, cfg)
#results_noise = run_sklearngp_noise(X_train, y_train, X_test, y_test, cfg)

print(results_no_noise)
#print(results_noise)