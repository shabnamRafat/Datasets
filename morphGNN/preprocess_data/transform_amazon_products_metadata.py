import pandas as pd
import json
import re
from collections import Counter
import os


# ============================================
# Directories & Filepaths
# ============================================

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'config.json'))
with open(config_path, 'r') as f:
    config = json.load(f)

datasets_path = config['datasets']

# ============================================================================
# Functions
# ============================================================================

def safe_get(d, key, default=''):
    """Safely get value from dict"""
    val = d.get(key, default)
    return val if val is not None else default


def join_list(lst, default=''):
    """Join list to string"""
    if not isinstance(lst, list):
        return default
    return ' '.join([str(x) for x in lst if x])


def extract_categories(cats):
    """Extract hierarchical category levels"""
    if not isinstance(cats, list) or len(cats) == 0:
        return None, None, None, 0

    level_0 = cats[0] if len(cats) > 0 else None
    level_1 = cats[1] if len(cats) > 1 else None
    level_2 = cats[2] if len(cats) > 2 else None
    depth = len(cats)

    return level_0, level_1, level_2, depth


def clean_text(text):
    """Basic text cleaning"""
    if not text:
        return ''
    text = str(text).lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


# ============================================================================
# MAIN TRANSFORMATION
# ============================================================================

def transform_product(row):
    """Transform single product record"""

    # Direct scalars
    price = row.get('price')
    if price is None or pd.isna(price):
        price = 0.0
    else:
        try:
            price = float(price)
        except (ValueError, TypeError):
            price = 0.0

    # Categories
    cats = row.get('categories', [])
    cat_0, cat_1, cat_2, cat_depth = extract_categories(cats)

    # Details dictionary
    details = row.get('details', {})
    if not isinstance(details, dict):
        details = {}

    # Text fields
    features = row.get('features', [])
    description = row.get('description', [])

    # Images
    images = row.get('images', [])
    if not isinstance(images, list):
        images = []

    # Build transformed record
    transformed = {
        # Identifiers
        'parent_asin': row.get('parent_asin'),

        # Scalars
        'price': price,
        'average_rating': float(row.get('average_rating', 0.0) or 0.0),
        'rating_number': int(row.get('rating_number', 0) or 0),
        'image_count': len(images),

        # Binary flags
        'has_price': 1 if price > 0 else 0,
        'has_description': 1 if description and len(description) > 0 else 0,
        'has_features': 1 if features and len(features) > 0 else 0,

        # Categorical
        'brand': clean_text(safe_get(details, 'Brand', 'unknown')),
        'store': clean_text(row.get('store', 'unknown')),
        'main_category': clean_text(row.get('main_category', 'unknown')),

        # Hierarchical categories
        'cat_level_0': clean_text(cat_0) if cat_0 else 'unknown',
        'cat_level_1': clean_text(cat_1) if cat_1 else 'unknown',
        'cat_level_2': clean_text(cat_2) if cat_2 else 'unknown',
        'category_depth': cat_depth,

        # Text for embeddings
        'title': clean_text(row.get('title', '')),
        'features_text': clean_text(join_list(features)),
        'description_text': clean_text(join_list(description)),

        # Details extracted
        'color': clean_text(safe_get(details, 'Color', 'unknown')),
        'material': clean_text(safe_get(details, 'Material', 'unknown')),
        'compatible_models': clean_text(safe_get(details, 'Compatible Phone Models', 'unknown')),
    }

    return transformed


def build_vocabulary(df, column, top_n=1000):
    """Build vocabulary mapping for categorical features"""
    counts = Counter(df[column].dropna())

    # Keep top N most common
    vocab = {'unknown': 0}  # 0 reserved for unknown
    for i, (value, count) in enumerate(counts.most_common(top_n), start=1):
        vocab[value] = i

    return vocab


def encode_categorical(df, column, vocab):
    """Encode categorical column using vocabulary"""
    return df[column].map(lambda x: vocab.get(x, 0))


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

def process_metadata(input_file, output_file):
    """Process metadata file and save transformed version"""
    print(f"\nProcessing: {input_file}")
    print("=" * 80)

    # Read data
    print("Loading data...")
    data = []

    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                transformed = transform_product(record)
                data.append(transformed)
            except json.JSONDecodeError as e:
                print(f"  Skipping malformed line {i}: {e}")
            if len(data) % 100000 == 0 and len(data) > 0:
                print(f"  Processed {len(data):,} products so far...", end='\r')

    print(f"\n  Total products: {len(data):,}")

    # Convert to DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame(data)

    # Build vocabularies
    print("Building vocabularies...")
    brand_vocab = build_vocabulary(df, 'brand', top_n=1000)
    store_vocab = build_vocabulary(df, 'store', top_n=500)
    cat0_vocab = build_vocabulary(df, 'cat_level_0', top_n=50)

    print(f"  Brand vocabulary: {len(brand_vocab):,} entries")
    print(f"  Store vocabulary: {len(store_vocab):,} entries")
    print(f"  Category L0 vocabulary: {len(cat0_vocab):,} entries")

    # Encode categoricals
    print("Encoding categorical features...")
    df['brand_id'] = encode_categorical(df, 'brand', brand_vocab)
    df['store_id'] = encode_categorical(df, 'store', store_vocab)
    df['main_category_id'] = encode_categorical(df, 'cat_level_0', cat0_vocab)

    # Save
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)

    # Save vocabularies
    vocab_file = output_file.replace('.csv', '_vocabs.json')
    print(f"Saving vocabularies to {vocab_file}...")
    vocabs = {
        'brand': brand_vocab,
        'store': store_vocab,
        'main_category': cat0_vocab
    }
    with open(vocab_file, 'w') as f:
        json.dump(vocabs, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("TRANSFORMATION COMPLETE")
    print("=" * 80)
    print(f"Total products: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nOutput files:")
    print(f"  Data: {output_file}")
    print(f"  Vocabularies: {vocab_file}")
    print(f"\nSample stats:")
    print(f"  Products with price: {df['has_price'].sum():,} ({df['has_price'].mean() * 100:.1f}%)")
    print(f"  Products with features: {df['has_features'].sum():,} ({df['has_features'].mean() * 100:.1f}%)")
    print(f"  Products with description: {df['has_description'].sum():,} ({df['has_description'].mean() * 100:.1f}%)")
    print(f"  Avg images per product: {df['image_count'].mean():.1f}")
    print("=" * 80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Process all metadata files"""

    files = [
        {
            'input': os.path.abspath(os.path.join(datasets_path, 'e-commerce/meta_amazon_gift_cards.jsonl')),
            'output': os.path.abspath(os.path.join(datasets_path, 'e-commerce/transformed_amazon_gift_cards_metadata.csv'))
        }
    ]

    print("\n" + "=" * 80)
    print("METADATA TRANSFORMATION PIPELINE")
    print("=" * 80)

    for file_config in files:
        process_metadata(file_config['input'], file_config['output'])

    print("\nALL TRANSFORMATIONS COMPLETE!")


if __name__ == "__main__":
    main()