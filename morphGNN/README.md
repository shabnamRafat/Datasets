# Graph Data Download and Preparation

This guide explains how to download and prepare all graph datasets used in "MorphGNN: Automated GNN Architecture Selection via Graph Property Profiling" project.

---

## Overview

Datasets come from four different sources, each with its own download script. After downloading, all datasets go through a common splitting step.

```
Download Scripts  →  Raw Graph PKL Files  →  graph_splitter.py  →  Train / Val / Test splits
```

---

## Datasets & Scripts

### 1. PyG Datasets — `download_graph_data_from_PyG.py`

**Covers:** Cora, PubMed, Coauthor, Wiki-CS, ACM, DBLP, IMDB, Freebase, MovieLens, Amazon Book

**Steps:**
1. Set `dataset_name` in the config to one of the supported values below
2. Run `download_graph_data_from_PyG.py`
3. The `.pkl` file is saved automatically to the graph data folder

**Supported values:**

| dataset_name | Dataset |
|---|---|
| `amazon_book` | Amazon Book |
| `cora` | Cora |
| `pubmed` | PubMed |
| `coauthor` | Coauthor CS |
| `wiki_cs` | Wiki-CS |
| `acm` | ACM |
| `dblp` | DBLP |
| `imdb` | IMDB |
| `freebase` | Freebase |
| `movielens` | MovieLens |

---

### 2. IGMC Datasets — `download_graph_data_from_IGMC.py`

**Covers:** IGMC Douban, IGMC Flixster, IGMC Yahoo Music

**Steps:**
1. Set `dataset_name` in the config to one of: `douban`, `flixster`, `yahoo_music`
2. Run `download_graph_data_from_IGMC.py`
3. The `.pkl` file is saved automatically to the graph data folder

---

### 3. JODIE Datasets — `download_graph_data_from_jodie.py`

**Covers:** JODIE Wikipedia, JODIE Reddit

**Source:** [SNAP Stanford](http://snap.stanford.edu/jodie/)

**Steps:**
1. Set `dataset_name` in the config to one of: `jodie_wikipedia`, `jodie_reddit`
2. Run `download_graph_data_from_jodie.py`
3. The script downloads the raw CSV from SNAP Stanford and saves the `.pkl` to the graph data folder

---

### 4. TGB Datasets — `download_graph_data_from_tgbl.py`

**Covers:** JODIE Wikipedia (alternative source via TGB)

**Steps:**
1. Install the TGB library:
```bash
pip install py-tgb
```
2. Set `dataset_name` in the config to `jodie_wikipedia`
3. Run `download_graph_data_from_tgbl.py`
4. The `.pkl` file is saved automatically to the graph data folder

---

### 5. Amazon Product Datasets — Custom Pipeline

**Covers:** Amazon Music Instruments, Amazon Gift Cards, Amazon Beauty

**Source:** [McAuley-Lab Amazon Reviews 2023 on HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw)

**Steps:**
1. Manually download the following files from HuggingFace and place them in the datasets folder:

| File | Type |
|---|---|
| `meta_Musical_Instruments.jsonl` | Product metadata |
| `meta_Gift_Cards.jsonl` | Product metadata |
| `meta_All_Beauty.jsonl` | Product metadata |
| `Musical_Instruments.jsonl` | Reviews |
| `Gift_Cards.jsonl` | Reviews |
| `All_Beauty.jsonl` | Reviews |

2. Run the following scripts in order:
```
transform_amazon_products_metadata.py
preprocess_amazon_products_metadata.py
preprocess_amazon_products_reviews.py
build_ecommerce_graph_pkl.py
```
3. The final `.pkl` is produced by `build_ecommerce_graph_pkl.py` and saved to the graph data folder

---

### 6. TripAdvisor Dataset — Custom Pipeline

**Covers:** TripAdvisor Restaurants

**Source:** [Kaggle — TripAdvisor Restaurant Recommendation Data USA](https://www.kaggle.com/datasets/siddharthmandgi/tripadvisor-restaurant-recommendation-data-usa)

**Steps:**
1. Download the CSV file from Kaggle and place it in the datasets folder
2. Run `build_restaurant_graph_pkl.py`
3. The final `.pkl` is saved to the graph data folder

---

## Splitting

After any dataset is downloaded and saved as a `.pkl`, run:

```bash
python graph_splitter.py
```

This splits the graph into train, validation, and test sets and saves labeled versions back to the `graph_datasets` folder.

---

## Dataset Summary

| Dataset | Script |
|---|---|
| Cora, PubMed, Coauthor, Wiki-CS | `download_graph_data_from_PyG.py` |
| ACM, DBLP, IMDB, Freebase | `download_graph_data_from_PyG.py` |
| MovieLens, Amazon Book | `download_graph_data_from_PyG.py` |
| IGMC Douban, Flixster, Yahoo Music | `download_graph_data_from_IGMC.py` |
| JODIE Wikipedia, Reddit | `download_graph_data_from_jodie.py` |
| Amazon Beauty, Gift Cards, Music Instruments | Custom Amazon pipeline |
| TripAdvisor | Custom TripAdvisor pipeline |
