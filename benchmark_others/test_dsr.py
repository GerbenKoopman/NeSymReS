from scripts.run_dsr import run_dsr#, run_dsr_noise
from scripts.bench_utils import generate_dataset
import pandas as pd
import yaml

df = pd.read_csv("test_set/nc.csv")

X, y, expr = generate_dataset(df.iloc[100])


with open("benchmark_others/configs/dsr.json") as f:
    cfg = yaml.safe_load(f)

results_no_noise = run_dsr(X, y, cfg)
#results_noise = run_gp_noise(X, y, cfg)

print(results_no_noise)
#print(results_noise)