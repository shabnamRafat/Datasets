import os
import json
import re
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder, StandardScaler

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'config.json'))
with open(config_path, 'r') as f:
    config = json.load(f)

datasets_path = config['datasets']
graph_datasets_path = config['graph_datasets']

DATA_FILE = os.path.join(datasets_path, 'trip_advisor_restaurant_dataset.csv')
OUTPUT_FILE = os.path.join(graph_datasets_path, 'trip_advisor_restaurant_graph.pkl')


def extract_rating(x):
    if pd.isna(x):
        return 0.0
    m = re.search(r'(\d+(\.\d+)?)', str(x))
    return float(m.group(1)) if m else 0.0


def extract_review_count(x):
    if pd.isna(x):
        return 0.0
    s = str(x).replace(',', '')
    m = re.search(r'(\d+)', s)
    return float(m.group(1)) if m else 0.0


def build_graph():
    df = pd.read_csv(DATA_FILE, sep=None, engine='python')
    df.columns = df.columns.str.strip()

    df['Location'] = df['Location'].astype(str).str.strip()
    df['Type'] = df['Type'].astype(str).str.strip()
    df['Price_Range'] = df['Price_Range'].astype(str).str.strip()

    df['rating'] = df['Reviews'].apply(extract_rating)
    df['num_reviews'] = df['No of Reviews'].apply(extract_review_count)

    num_cols = ['rating', 'num_reviews']
    df[num_cols] = df[num_cols].fillna(0)

    location_encoder = LabelEncoder()
    restaurant_type_encoder = LabelEncoder()
    price_encoder = LabelEncoder()

    df['location_idx'] = location_encoder.fit_transform(df['Location'])
    df['restaurant_type_idx'] = restaurant_type_encoder.fit_transform(df['Type'])
    df['price_idx'] = price_encoder.fit_transform(df['Price_Range'])

    scaler = StandardScaler()
    restaurant_x = torch.tensor(scaler.fit_transform(df[num_cols].values), dtype=torch.float)

    num_restaurants = len(df)
    num_locations = len(location_encoder.classes_)
    num_restaurant_types = len(restaurant_type_encoder.classes_)

    restaurant_idx = torch.arange(num_restaurants, dtype=torch.long)

    restaurant_location_edge = torch.stack([
        restaurant_idx,
        torch.tensor(df['location_idx'].values, dtype=torch.long)
    ], dim=0)

    restaurant_type_edge = torch.stack([
        restaurant_idx,
        torch.tensor(df['restaurant_type_idx'].values, dtype=torch.long)
    ], dim=0)

    data = HeteroData()

    data['restaurant'].x = restaurant_x
    data['restaurant'].y = torch.tensor(df['price_idx'].values, dtype=torch.long)

    data['location'].x = torch.eye(num_locations, dtype=torch.float)
    data['restaurant_type'].x = torch.eye(num_restaurant_types, dtype=torch.float)

    data['restaurant', 'in_location', 'location'].edge_index = restaurant_location_edge
    data['location', 'rev_in_location', 'restaurant'].edge_index = restaurant_location_edge.flip(0)

    data['restaurant', 'has_restaurant_type', 'restaurant_type'].edge_index = restaurant_type_edge
    data['restaurant_type', 'rev_has_restaurant_type', 'restaurant'].edge_index = restaurant_type_edge.flip(0)

    data['restaurant'].num_nodes = num_restaurants
    data['location'].num_nodes = num_locations
    data['restaurant_type'].num_nodes = num_restaurant_types

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    torch.save({
        'graph': data,
        'location_encoder': location_encoder,
        'restaurant_type_encoder': restaurant_type_encoder,
        'price_encoder': price_encoder,
        'scaler': scaler
    }, OUTPUT_FILE)


if __name__ == "__main__":
    build_graph()