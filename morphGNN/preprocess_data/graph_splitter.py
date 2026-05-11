"""
graph_splitter.py
"""

from __future__ import annotations

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../", "config.json")
)
with open(_config_path, "r") as _f:
    _cfg = json.load(_f)

_GRAPH_FILE        = "trip_advisor_restaurant_graph.pkl"

_input_path = os.path.abspath(
    os.path.join(
        os.path.dirname(_config_path),
        "src",
        _cfg["graph_datasets"],
        _GRAPH_FILE,
    )
)

SPLIT_CONFIG = {
    "input_path": _input_path,
    "train": 0.80,
    "val":   0.10,
    "test":  0.10,
    "strategy": "auto",
    "seed": 42,
}

_TEMPORAL_KEYS  = {"edge_time", "timestamps", "t", "time", "ts", "edge_ts"}
_EDGE_TYPE_KEYS = {"edge_type", "rel", "relation", "edge_reltype"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_tensor(val) -> torch.Tensor:
    """convert numpy array, list, or tensor to float tensor."""
    if isinstance(val, torch.Tensor):
        return val.float()
    return torch.tensor(val, dtype=torch.float32)


def _edge_index_from_dict(obj: dict) -> torch.Tensor:
    """extract edge_index from any known dict format."""
    if "edge_index" in obj:
        ei = obj["edge_index"]
        return ei if isinstance(ei, torch.Tensor) else torch.tensor(ei)
    if "edges" in obj:
        ei = obj["edges"]
        return ei if isinstance(ei, torch.Tensor) else torch.tensor(ei)
    if "sources" in obj and "destinations" in obj:
        src = _to_tensor(obj["sources"]).long()
        dst = _to_tensor(obj["destinations"]).long()
        return torch.stack([src, dst], dim=0)
    raise ValueError(
        "dict graph has no recognizable edge key — expected one of: "
        "edge_index, edges, sources+destinations"
    )


# ── loader ────────────────────────────────────────────────────────────────────

def load_any_graph(path: str | Path) -> Any:
    path   = Path(path)
    suffix = path.suffix.lower()
    errors = {}
    obj    = None

    if suffix == ".bin":
        try:
            import dgl
            graphs, _ = dgl.load_graphs(str(path))
            obj = graphs[0] if len(graphs) == 1 else graphs
            log.info("loaded via dgl.load_graphs: %s", path)
        except Exception as e:
            errors["dgl"] = e

    if obj is None:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            log.info("loaded via torch.load: %s", path)
        except Exception as e:
            errors["torch"] = e

    if obj is None:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            log.info("loaded via pickle.load: %s", path)
        except Exception as e:
            errors["pickle"] = e

    if obj is None:
        try:
            import joblib
            obj = joblib.load(path)
            log.info("loaded via joblib.load: %s", path)
        except Exception as e:
            errors["joblib"] = e

    if obj is None and suffix == ".npz":
        try:
            import scipy.sparse as sp
            obj = sp.load_npz(str(path))
            log.info("loaded via scipy.sparse.load_npz: %s", path)
        except Exception as e:
            errors["npz"] = e

    if obj is None and suffix in (".graphml", ".gml", ".gexf", ".xml"):
        try:
            import networkx as nx
            loaders = {
                ".graphml": nx.read_graphml,
                ".gml":     nx.read_gml,
                ".gexf":    nx.read_gexf,
                ".xml":     nx.read_graphml,
            }
            obj = loaders[suffix](str(path))
            log.info("loaded via networkx %s reader: %s", suffix, path)
        except Exception as e:
            errors["networkx"] = e

    if obj is None:
        raise RuntimeError(
            "all loaders failed for {}:\n{}".format(
                path, "\n".join(f"  {k}: {v}" for k, v in errors.items())
            )
        )

    if isinstance(obj, dict):
        for key, val in obj.items():
            if type(val).__name__ in ("Data", "HeteroData", "TemporalData", "DGLGraph", "DGLHeteroGraph"):
                log.info("unwrapping dict key '%s'", key)
                return val
        return obj

    log.info("loaded graph type: %s", type(obj).__name__)
    return obj


def save_graph(obj: Any, path: Path) -> None:
    if path.suffix.lower() == ".pt":
        torch.save(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    log.info("saved enriched graph → %s", path)


# ── strategy detection ────────────────────────────────────────────────────────

def detect_strategy(obj: Any) -> str:
    type_name = type(obj).__name__

    # TemporalData is always temporal
    if type_name == "TemporalData":
        log.info("strategy detected: temporal (TemporalData type)")
        return "temporal"

    if type_name == "HeteroData":
        for et in obj.edge_types:
            store = obj[et]
            for key in _TEMPORAL_KEYS:
                if store.get(key, None) is not None:
                    log.info("strategy detected: temporal (HeteroData has edge timestamps)")
                    return "temporal"
        node_types = [str(nt).lower() for nt in obj.node_types]
        if any("user" in nt or "item" in nt or "song" in nt or "track" in nt
               or "customer" in nt or "product" in nt
               for nt in node_types):
            log.info("strategy detected: edge (HeteroData has user/item node types)")
            return "edge"
        for nt in obj.node_types:
            if obj[nt].get("y", None) is not None:
                log.info("strategy detected: node (HeteroData has node labels)")
                return "node"
        log.info("strategy detected: edge (HeteroData fallback)")
        return "edge"

    if type_name == "Data":
        for key in _TEMPORAL_KEYS:
            if getattr(obj, key, None) is not None:
                log.info("strategy detected: temporal (Data has edge timestamps)")
                return "temporal"
        if getattr(obj, "num_users", 0) or getattr(obj, "num_items", 0):
            log.info("strategy detected: edge (Data has num_users/num_items)")
            return "edge"
        if getattr(obj, "y", None) is not None:
            log.info("strategy detected: node (Data has node labels)")
            return "node"
        log.info("strategy detected: edge (Data fallback)")
        return "edge"

    if type_name in ("DGLGraph", "DGLHeteroGraph"):
        for key in _TEMPORAL_KEYS:
            try:
                obj.edata[key]
                log.info("strategy detected: temporal (DGL has edge timestamps)")
                return "temporal"
            except KeyError:
                pass
        try:
            obj.ndata["label"]
            log.info("strategy detected: node (DGL has node labels)")
            return "node"
        except KeyError:
            pass
        log.info("strategy detected: edge (DGL fallback)")
        return "edge"

    if isinstance(obj, dict):
        for key in _TEMPORAL_KEYS:
            if key in obj:
                log.info("strategy detected: temporal (dict has timestamps key '%s')", key)
                return "temporal"
        if "num_users" in obj or "num_items" in obj:
            log.info("strategy detected: edge (dict has num_users/num_items)")
            return "edge"
        if "y" in obj or "label" in obj or "labels" in obj:
            log.info("strategy detected: node (dict has labels)")
            return "node"
        log.info("strategy detected: edge (dict fallback)")
        return "edge"

    try:
        import networkx as nx
        if isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            nodes = list(obj.nodes())
            if nodes:
                sample = dict(obj.nodes[nodes[0]])
                if "y" in sample or "label" in sample:
                    log.info("strategy detected: node (NetworkX has node labels)")
                    return "node"
            log.info("strategy detected: edge (NetworkX fallback)")
            return "edge"
    except ImportError:
        pass

    log.info("strategy detected: edge (final fallback)")
    return "edge"


# ── split helpers ─────────────────────────────────────────────────────────────

def _split_indices(n, train_ratio, val_ratio, seed):
    torch.manual_seed(seed)
    perm    = torch.randperm(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def _temporal_split_indices(timestamps, train_ratio, val_ratio):
    order   = timestamps.argsort()
    n       = len(order)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def _masks_from_indices(n, train_idx, val_idx, test_idx):
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True
    return train_mask, val_mask, test_mask


# ── node split ────────────────────────────────────────────────────────────────

def split_node(obj: Any, cfg: dict) -> tuple[Any, dict]:
    train_r = cfg["train"]
    val_r   = cfg["val"]
    seed    = cfg["seed"]
    report  = {"strategy": "node"}
    type_name = type(obj).__name__

    if type_name == "HeteroData":
        report["node_types"] = {}
        for nt in obj.node_types:
            store = obj[nt]
            n     = store.num_nodes
            ti, vi, sti = _split_indices(n, train_r, val_r, seed)
            tm, vm, stm = _masks_from_indices(n, ti, vi, sti)
            store.train_mask = tm
            store.val_mask   = vm
            store.test_mask  = stm
            report["node_types"][str(nt)] = {
                "total": n, "train": int(tm.sum()), "val": int(vm.sum()), "test": int(stm.sum())
            }
            log.info("  [%s] nodes → train=%d val=%d test=%d", nt, tm.sum(), vm.sum(), stm.sum())

    elif type_name in ("Data", "TemporalData"):
        n = obj.num_nodes
        ti, vi, sti = _split_indices(n, train_r, val_r, seed)
        tm, vm, stm = _masks_from_indices(n, ti, vi, sti)
        obj.train_mask = tm
        obj.val_mask   = vm
        obj.test_mask  = stm
        report["nodes"] = {
            "total": n, "train": int(tm.sum()), "val": int(vm.sum()), "test": int(stm.sum())
        }
        log.info("nodes → train=%d val=%d test=%d", tm.sum(), vm.sum(), stm.sum())

    elif type_name in ("DGLGraph", "DGLHeteroGraph"):
        n = obj.num_nodes()
        ti, vi, sti = _split_indices(n, train_r, val_r, seed)
        tm, vm, stm = _masks_from_indices(n, ti, vi, sti)
        obj.ndata["train_mask"] = tm
        obj.ndata["val_mask"]   = vm
        obj.ndata["test_mask"]  = stm
        report["nodes"] = {
            "total": n, "train": int(tm.sum()), "val": int(vm.sum()), "test": int(stm.sum())
        }
        log.info("nodes → train=%d val=%d test=%d", tm.sum(), vm.sum(), stm.sum())

    elif isinstance(obj, dict):
        x = obj.get("x", obj.get("node_feat", None))
        n = x.shape[0] if x is not None else obj.get("num_nodes", 0)
        ti, vi, sti = _split_indices(n, train_r, val_r, seed)
        tm, vm, stm = _masks_from_indices(n, ti, vi, sti)
        obj["train_mask"] = tm
        obj["val_mask"]   = vm
        obj["test_mask"]  = stm
        report["nodes"] = {
            "total": n, "train": int(tm.sum()), "val": int(vm.sum()), "test": int(stm.sum())
        }
        log.info("nodes → train=%d val=%d test=%d", tm.sum(), vm.sum(), stm.sum())

    else:
        raise TypeError(f"node split not supported for type: {type_name}")

    return obj, report


# ── edge split ────────────────────────────────────────────────────────────────

def split_edge(obj: Any, cfg: dict) -> tuple[Any, dict]:
    train_r = cfg["train"]
    val_r   = cfg["val"]
    seed    = cfg["seed"]
    report  = {"strategy": "edge"}
    type_name = type(obj).__name__

    if type_name == "HeteroData":
        report["edge_types"] = {}
        for et in obj.edge_types:
            store = obj[et]
            if not hasattr(store, "edge_index"):
                continue
            ei = store.edge_index
            n  = ei.shape[1]
            ti, vi, sti = _split_indices(n, train_r, val_r, seed)
            store.full_edge_index  = ei.clone()
            store.train_edge_index = ei[:, ti]
            store.val_edge_index   = ei[:, vi]
            store.test_edge_index  = ei[:, sti]
            store.edge_index       = ei[:, ti]
            report["edge_types"][str(et)] = {
                "total": n, "train": ti.shape[0], "val": vi.shape[0], "test": sti.shape[0]
            }
            log.info("  [%s] edges → train=%d val=%d test=%d", et, ti.shape[0], vi.shape[0], sti.shape[0])

    elif type_name in ("Data", "TemporalData"):
        ei = obj.edge_index
        n  = ei.shape[1]
        ti, vi, sti = _split_indices(n, train_r, val_r, seed)
        obj.full_edge_index  = ei.clone()
        obj.train_edge_index = ei[:, ti]
        obj.val_edge_index   = ei[:, vi]
        obj.test_edge_index  = ei[:, sti]
        obj.edge_index       = ei[:, ti]
        report["edges"] = {
            "total": n, "train": ti.shape[0], "val": vi.shape[0], "test": sti.shape[0]
        }
        log.info("edges → train=%d val=%d test=%d", ti.shape[0], vi.shape[0], sti.shape[0])

    elif type_name in ("DGLGraph", "DGLHeteroGraph"):
        n = obj.num_edges()
        ti, vi, sti = _split_indices(n, train_r, val_r, seed)
        tm, vm, stm = _masks_from_indices(n, ti, vi, sti)
        obj.edata["train_mask"] = tm
        obj.edata["val_mask"]   = vm
        obj.edata["test_mask"]  = stm
        report["edges"] = {
            "total": n, "train": int(tm.sum()), "val": int(vm.sum()), "test": int(stm.sum())
        }
        log.info("edges → train=%d val=%d test=%d", tm.sum(), vm.sum(), stm.sum())

    elif isinstance(obj, dict):
        ei = _edge_index_from_dict(obj)
        n  = ei.shape[1]
        ti, vi, sti = _split_indices(n, train_r, val_r, seed)
        obj["full_edge_index"]  = ei.clone()
        obj["train_edge_index"] = ei[:, ti]
        obj["val_edge_index"]   = ei[:, vi]
        obj["test_edge_index"]  = ei[:, sti]
        obj["edge_index"]       = ei[:, ti]
        report["edges"] = {
            "total": n, "train": ti.shape[0], "val": vi.shape[0], "test": sti.shape[0]
        }
        log.info("edges → train=%d val=%d test=%d", ti.shape[0], vi.shape[0], sti.shape[0])

    else:
        raise TypeError(f"edge split not supported for type: {type_name}")

    return obj, report


# ── temporal split ────────────────────────────────────────────────────────────

def split_temporal(obj: Any, cfg: dict) -> tuple[Any, dict]:
    train_r   = cfg["train"]
    val_r     = cfg["val"]
    report    = {"strategy": "temporal"}
    type_name = type(obj).__name__

    def _find_ts(store, getter) -> torch.Tensor | None:
        for key in _TEMPORAL_KEYS:
            val = getter(store, key)
            if val is not None:
                return _to_tensor(val)
        return None

    def _apply(ei, ts, n, seed):
        if ts is not None and len(ts) == n:
            return _temporal_split_indices(ts, train_r, val_r)
        log.warning("no valid timestamps — falling back to random edge split")
        return _split_indices(n, train_r, val_r, seed)

    if type_name == "HeteroData":
        report["edge_types"] = {}
        for et in obj.edge_types:
            store = obj[et]
            ts    = _find_ts(store, lambda s, k: s.get(k, None))
            ei    = store.get("edge_index", None)
            if ei is None:
                continue
            n  = ei.shape[1]
            ti, vi, sti = _apply(ei, ts, n, cfg["seed"])
            store.full_edge_index  = ei.clone()
            store.train_edge_index = ei[:, ti]
            store.val_edge_index   = ei[:, vi]
            store.test_edge_index  = ei[:, sti]
            store.edge_index       = ei[:, ti]
            if ts is not None:
                store.train_edge_time = ts[ti]
                store.val_edge_time   = ts[vi]
                store.test_edge_time  = ts[sti]
            report["edge_types"][str(et)] = {
                "total": n, "train": ti.shape[0], "val": vi.shape[0], "test": sti.shape[0],
                "time_ordered": ts is not None,
            }
            log.info("  [%s] temporal edges → train=%d val=%d test=%d", et, ti.shape[0], vi.shape[0], sti.shape[0])

    elif type_name in ("Data", "TemporalData"):
        ts = _find_ts(obj, lambda s, k: getattr(s, k, None))
        ei = obj.edge_index
        n  = ei.shape[1]
        ti, vi, sti = _apply(ei, ts, n, cfg["seed"])
        obj.full_edge_index  = ei.clone()
        obj.train_edge_index = ei[:, ti]
        obj.val_edge_index   = ei[:, vi]
        obj.test_edge_index  = ei[:, sti]
        obj.edge_index       = ei[:, ti]
        if ts is not None:
            obj.train_edge_time = ts[ti]
            obj.val_edge_time   = ts[vi]
            obj.test_edge_time  = ts[sti]
        report["edges"] = {
            "total": n, "train": ti.shape[0], "val": vi.shape[0], "test": sti.shape[0],
            "time_ordered": ts is not None,
        }
        log.info("temporal edges → train=%d val=%d test=%d", ti.shape[0], vi.shape[0], sti.shape[0])

    elif type_name in ("DGLGraph", "DGLHeteroGraph"):
        ts = None
        for key in _TEMPORAL_KEYS:
            try:
                ts = _to_tensor(obj.edata[key])
                break
            except KeyError:
                pass
        n = obj.num_edges()
        ti, vi, sti = _apply(None, ts, n, cfg["seed"])
        tm, vm, stm = _masks_from_indices(n, ti, vi, sti)
        obj.edata["train_mask"] = tm
        obj.edata["val_mask"]   = vm
        obj.edata["test_mask"]  = stm
        if ts is not None:
            obj.edata["train_edge_time"] = ts[ti]
            obj.edata["val_edge_time"]   = ts[vi]
            obj.edata["test_edge_time"]  = ts[sti]
        report["edges"] = {
            "total": n, "train": int(tm.sum()), "val": int(vm.sum()), "test": int(stm.sum()),
            "time_ordered": ts is not None,
        }
        log.info("temporal edges → train=%d val=%d test=%d", tm.sum(), vm.sum(), stm.sum())

    elif isinstance(obj, dict):
        ts = None
        for key in _TEMPORAL_KEYS:
            if key in obj:
                ts = _to_tensor(obj[key])
                break
        ei = _edge_index_from_dict(obj)
        n  = ei.shape[1]
        ti, vi, sti = _apply(ei, ts, n, cfg["seed"])
        obj["full_edge_index"]  = ei.clone()
        obj["train_edge_index"] = ei[:, ti]
        obj["val_edge_index"]   = ei[:, vi]
        obj["test_edge_index"]  = ei[:, sti]
        obj["edge_index"]       = ei[:, ti]
        if ts is not None:
            obj["train_edge_time"] = ts[ti]
            obj["val_edge_time"]   = ts[vi]
            obj["test_edge_time"]  = ts[sti]
        report["edges"] = {
            "total": n, "train": ti.shape[0], "val": vi.shape[0], "test": sti.shape[0],
            "time_ordered": ts is not None,
        }
        log.info("temporal edges → train=%d val=%d test=%d", ti.shape[0], vi.shape[0], sti.shape[0])

    else:
        raise TypeError(f"temporal split not supported for type: {type_name}")

    return obj, report


# ── validation ────────────────────────────────────────────────────────────────

def validate_config(cfg: dict) -> None:
    total = cfg["train"] + cfg["val"] + cfg["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0 — got {cfg['train']} + {cfg['val']} + {cfg['test']} = {total:.4f}"
        )
    for key in ("train", "val", "test"):
        if not (0.0 < cfg[key] < 1.0):
            raise ValueError(f"split ratio '{key}' must be between 0 and 1 — got {cfg[key]}")
    valid_strategies = {"auto", "node", "edge", "temporal"}
    if cfg["strategy"] not in valid_strategies:
        raise ValueError(f"unknown strategy '{cfg['strategy']}'. choose from: {valid_strategies}")
    log.info("config valid: train=%.0f%% val=%.0f%% test=%.0f%% strategy=%s",
             cfg["train"] * 100, cfg["val"] * 100, cfg["test"] * 100, cfg["strategy"])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = SPLIT_CONFIG
    validate_config(cfg)

    input_path  = Path(cfg["input_path"])
    output_path = input_path.parent / (input_path.stem + "_split" + input_path.suffix)
    report_path = input_path.parent / (input_path.stem + "_split_report.json")

    log.info("loading graph from %s", input_path)
    obj = load_any_graph(input_path)

    strategy = cfg["strategy"]
    if strategy == "auto":
        strategy = detect_strategy(obj)
    log.info("using strategy: %s", strategy)

    splitters = {
        "node":     split_node,
        "edge":     split_edge,
        "temporal": split_temporal,
    }
    obj, report = splitters[strategy](obj, cfg)

    report["config"] = {
        "input_path":  str(input_path),
        "output_path": str(output_path),
        "train":       cfg["train"],
        "val":         cfg["val"],
        "test":        cfg["test"],
        "seed":        cfg["seed"],
        "strategy":    strategy,
    }

    save_graph(obj, output_path)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("split report saved → %s", report_path)
    log.info("done — output: %s", output_path)


if __name__ == "__main__":
    main()