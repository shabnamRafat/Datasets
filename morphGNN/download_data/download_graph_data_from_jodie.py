import os
import urllib.request
import pickle
import pandas as pd
import torch
from torch_geometric.data import TemporalData
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'config.json'))
with open(config_path, 'r') as f:
    config = json.load(f)

datasets_path = config['graph_datasets']
dataset_name = config['dataset_name']
os.makedirs(datasets_path, exist_ok=True)

url = "http://snap.stanford.edu/jodie/reddit.csv"
csv_path = os.path.join(datasets_path, f"{dataset_name}.csv")

urllib.request.urlretrieve(url, csv_path)

df = pd.read_csv(csv_path, header=None, skiprows=1)

src      = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
dst      = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
t        = torch.tensor(df.iloc[:, 2].values, dtype=torch.float)
label    = torch.tensor(df.iloc[:, 3].values, dtype=torch.float)
features = torch.tensor(df.iloc[:, 4:].values, dtype=torch.float)

data = TemporalData(src=src, dst=dst, t=t, y=label, msg=features)

with open(os.path.join(datasets_path, f"{dataset_name}_graph.pkl"), "wb") as f:
    pickle.dump(data, f)