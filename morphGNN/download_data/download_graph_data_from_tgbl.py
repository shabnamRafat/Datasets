from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import pickle
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
tgbl_map = config['tgbl_map']
os.makedirs(datasets_path, exist_ok=True)

tgbl_name = tgbl_map[dataset_name]
dataset = PyGLinkPropPredDataset(name=tgbl_name, root=datasets_path)
data = dataset.get_TemporalData()

with open(os.path.join(datasets_path, f"{dataset_name}_graph.pkl"), "wb") as f:
    pickle.dump(data, f)