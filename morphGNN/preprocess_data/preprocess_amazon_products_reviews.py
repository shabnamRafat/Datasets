import pandas as pd
import os
import json


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'config.json'))
with open(config_path, 'r') as f:
    config = json.load(f)

datasets_path = config['datasets']

# Load metadata (cleaned)
print("Loading metadata...")
metadata = pd.read_csv(os.path.abspath(os.path.join(datasets_path, 'e-commerce/amazon_gift_cards_metadata_cleaned.csv')))
valid_items = set(metadata['parent_asin'].unique())
print(f"Valid items in metadata: {len(valid_items):,}")

# Load reviews
print("Loading and filtering reviews...")
reviews_path = os.path.abspath(os.path.join(datasets_path, 'e-commerce/review_amazon_gift_cards.jsonl'))
COLUMNS = ['user_id', 'parent_asin', 'rating', 'timestamp', 'title', 'text']

records = []
skipped = 0
with open(reviews_path, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get('parent_asin') in valid_items:
                records.append({col: record.get(col) for col in COLUMNS})
        except json.JSONDecodeError as e:
            skipped += 1
            print(f"  Skipping malformed line {i}: {e}")
        if (i + 1) % 500000 == 0:
            print(f"  Processed {i+1:,} lines, {len(records):,} kept so far...")

print(f"\nTotal reviews after filter: {len(records):,}")
if skipped:
    print(f"Skipped malformed lines: {skipped}")

reviews_filtered = pd.DataFrame(records, columns=COLUMNS)

# Save
output_path = os.path.abspath(os.path.join(datasets_path, 'e-commerce/amazon_gift_cards_reviews_cleaned.csv'))
reviews_filtered.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")