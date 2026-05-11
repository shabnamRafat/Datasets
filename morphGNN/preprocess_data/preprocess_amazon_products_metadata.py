import pandas as pd
import json
import os


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'config.json'))
with open(config_path, 'r') as f:
    config = json.load(f)



df = pd.read_csv(os.path.abspath(os.path.join(config['datasets'], 'e-commerce/transformed_amazon_gift_cards_metadata.csv')))
df = df[df['main_category'] == 'gift cards']



fields_to_remove = ['category_depth', 'store', 'store_id', 'has_price', 'has_features', 'has_description', 'main_category', 'main_category_id', 'cat_level_0']
df = df.drop(columns=fields_to_remove)

df.to_csv(os.path.abspath(os.path.join(config['datasets'],'e-commerce/amazon_gift_cards_metadata_cleaned.csv')), index=False)