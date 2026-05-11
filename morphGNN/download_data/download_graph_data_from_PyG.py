from torch_geometric.datasets import AmazonBook, Planetoid, Coauthor, WikiCS, HGBDataset, MovieLens
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
dataset_map = config['dataset_map']
os.makedirs(datasets_path, exist_ok=True)

CLASS_REGISTRY = {
    "AmazonBook": AmazonBook,
    "Planetoid":  Planetoid,
    "Coauthor":   Coauthor,
    "WikiCS":     WikiCS,
    "HGBDataset": HGBDataset,
    "MovieLens":  MovieLens,
}

try:
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Must be one of: {list(dataset_map.keys())}")

    entry = dataset_map[dataset_name]
    cls = CLASS_REGISTRY[entry['class']]
    dataset = cls(root=datasets_path, **entry['args'])
    data = dataset[0]

    with open(os.path.join(datasets_path, f"{dataset_name}_graph.pkl"), "wb") as f:
        pickle.dump(data, f)

except urllib.error.HTTPError as e:
    if e.code == 404:
        print("Download failed because the upstream PyG URL is broken.")
    else:
        raise