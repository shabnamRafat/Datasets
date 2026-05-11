from torch_geometric.datasets import IGMCDataset
import pickle
import urllib.error
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'config.json'))
with open(config_path, 'r') as f:
    config = json.load(f)

datasets_path = config['graph_datasets']
dataset_name = config['dataset_name']
os.makedirs(datasets_path, exist_ok=True)

try:
    dataset = IGMCDataset(root=datasets_path, name=dataset_name)
    data = dataset[0]

    with open(os.path.join(datasets_path, f"{dataset_name}_graph.pkl"), "wb") as f:
        pickle.dump(data, f)

except urllib.error.HTTPError as e:
    if e.code == 404:
        print("Download failed because the upstream URL is broken.")
    else:
        raise