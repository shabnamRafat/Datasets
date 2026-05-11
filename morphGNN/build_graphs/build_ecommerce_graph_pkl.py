import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import json
import tempfile
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '../', 'config.json'))
with open(config_path, 'r') as f:
    config = json.load(f)

datasets_path = config['datasets']
graph_datasets_path = config['graph_datasets']

REVIEWS_FILE = os.path.abspath(os.path.join(datasets_path, 'amazon_gift_cards_reviews_cleaned.csv'))
METADATA_FILE = os.path.abspath(os.path.join(datasets_path, 'amazon_gift_cards_metadata_cleaned.csv'))
OUTPUT_FILE = os.path.abspath(os.path.join(graph_datasets_path, 'amazon_gift_cards_graph.pkl'))

EMBEDDING_MODEL = 'all-mpnet-base-v2'  # 768 dims


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_embeddings_in_chunks(texts, model, batch_size=512, desc="Embedding"):
    """Generate BERT embeddings in chunks and stream to disk to avoid OOM"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{desc}...")
    print(f"  Total texts: {len(texts):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Streaming to disk every 10,000 embeddings...")

    # Create temp directory for chunks
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    current_chunk = []
    chunk_idx = 0
    CHUNK_SAVE_SIZE = 10000  # Save to disk every 10K embeddings

    with tqdm(range(0, len(texts), batch_size), desc=f"  {desc}", ncols=80) as pbar:
        for i in pbar:
            batch = texts[i:i + batch_size]

            # Generate embeddings for this batch
            batch_emb = model.encode(
                batch,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=False,
                show_progress_bar=False
            )

            # Accumulate on CPU
            current_chunk.append(batch_emb.cpu())

            # Save chunk to disk when accumulated enough
            if len(current_chunk) * batch_size >= CHUNK_SAVE_SIZE or i + batch_size >= len(texts):
                # Concatenate accumulated batches
                chunk_tensor = torch.cat(current_chunk, dim=0)

                # Save to disk
                chunk_file = os.path.join(temp_dir, f'emb_chunk_{chunk_idx}.pt')
                torch.save(chunk_tensor, chunk_file)
                chunk_files.append(chunk_file)

                pbar.set_postfix_str(f'saved chunk {chunk_idx}')

                # Clear memory
                current_chunk = []
                chunk_idx += 1
                del chunk_tensor, batch_emb
                if device == 'cuda':
                    torch.cuda.empty_cache()

    # Load all chunks from disk and combine
    print(f"  Loading {len(chunk_files)} chunks from disk...")
    embeddings = []

    for chunk_file in tqdm(chunk_files, desc=f"  Loading {desc}", ncols=80):
        chunk = torch.load(chunk_file)
        embeddings.append(chunk)
        os.remove(chunk_file)  # Delete immediately after loading

    os.rmdir(temp_dir)

    final_embeddings = torch.cat(embeddings, dim=0)
    print(f"  ✓ Final shape: {final_embeddings.shape}")

    return final_embeddings


def build_user_profile_features(reviews_df):
    """Build user profile features from review history"""
    print("\nBuilding user profile features...")

    user_profiles = reviews_df.groupby('user_id').agg({
        'rating': ['mean', 'std', 'count'],
        'timestamp': ['min', 'max']
    }).reset_index()

    user_profiles.columns = ['user_id', 'avg_rating', 'rating_std', 'review_count', 'first_review', 'last_review']

    # Fill NaN std (users with only 1 review)
    user_profiles['rating_std'] = user_profiles['rating_std'].fillna(0)

    # Compute user activity span (days)
    user_profiles['first_review'] = pd.to_datetime(user_profiles['first_review'])
    user_profiles['last_review'] = pd.to_datetime(user_profiles['last_review'])
    user_profiles['activity_span_days'] = (user_profiles['last_review'] - user_profiles['first_review']).dt.days

    # Normalize features
    scaler = StandardScaler()
    user_profiles[
        ['avg_rating_norm', 'rating_std_norm', 'review_count_norm', 'activity_span_norm']] = scaler.fit_transform(
        user_profiles[['avg_rating', 'rating_std', 'review_count', 'activity_span_days']]
    )

    print(f"  Built profiles for {len(user_profiles):,} users")

    return user_profiles


def build_item_features(metadata_df, embedding_model):
    """Build item features from metadata"""
    print("\nBuilding item features...")

    # Prepare text fields (fill empty strings)
    metadata_df['title'] = metadata_df['title'].fillna('').astype(str)
    metadata_df['features_text'] = metadata_df['features_text'].fillna('').astype(str)
    metadata_df['description_text'] = metadata_df['description_text'].fillna('').astype(str)

    # Generate embeddings with streaming
    title_embeddings = generate_embeddings_in_chunks(
        metadata_df['title'].tolist(),
        embedding_model,
        batch_size=512,
        desc="Title embeddings"
    )

    features_embeddings = generate_embeddings_in_chunks(
        metadata_df['features_text'].tolist(),
        embedding_model,
        batch_size=512,
        desc="Feature embeddings"
    )

    description_embeddings = generate_embeddings_in_chunks(
        metadata_df['description_text'].tolist(),
        embedding_model,
        batch_size=512,
        desc="Description embeddings"
    )

    # ========================================================================
    # NUMERICAL FEATURES
    # ========================================================================

    print("\nBuilding numerical features...")
    numerical_cols = ['price', 'average_rating', 'rating_number', 'image_count']
    numerical_features = metadata_df[numerical_cols].fillna(0).values

    # Normalize
    scaler = StandardScaler()
    numerical_features_norm = scaler.fit_transform(numerical_features)
    numerical_features_tensor = torch.tensor(numerical_features_norm, dtype=torch.float)

    # ========================================================================
    # CATEGORICAL FEATURES
    # ========================================================================

    categorical_cols = ['brand_id']
    categorical_features = torch.tensor(metadata_df[categorical_cols].values, dtype=torch.long)

    # ========================================================================
    # COMBINE ALL FEATURES
    # ========================================================================

    print("\nCombining features...")

    # Combine in stages to avoid OOM
    print("  Stage 1: Title + Features...")
    combined_text = torch.cat([title_embeddings, features_embeddings], dim=1)
    del title_embeddings, features_embeddings  # Free 6GB immediately
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Stage 2: + Description...")
    combined_text = torch.cat([combined_text, description_embeddings], dim=1)
    del description_embeddings  # Free 3GB
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Stage 3: + Numerical + Categorical...")
    item_features = torch.cat([
        combined_text,  # 2304 dims (768×3)
        numerical_features_tensor,  # 4 dims
        categorical_features.float()  # 1 dim
    ], dim=1)
    del combined_text  # Free intermediate tensor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"  ✓ Item features shape: {item_features.shape}")
    print(f"    - Text embeddings: 2304 dims (title + features + description)")
    print(f"    - Numerical: 4 dims")
    print(f"    - Categorical: 1 dim")
    print(f"    - Total: {item_features.shape[1]} dims")

    return item_features


def build_edges(reviews_df, user_encoder, item_encoder):
    """Build edge index and edge attributes"""
    print("\nBuilding edges...")

    # Encode user and item IDs
    reviews_df['user_idx'] = user_encoder.transform(reviews_df['user_id'])
    reviews_df['item_idx'] = item_encoder.transform(reviews_df['parent_asin'])

    # Edge index (user -> item)
    edge_index = torch.tensor(
        reviews_df[['user_idx', 'item_idx']].values.T,
        dtype=torch.long
    )

    # Edge weights (normalized ratings)
    ratings = reviews_df['rating'].values
    edge_weight = torch.tensor(ratings / 5.0, dtype=torch.float)

    # Edge attributes (raw ratings)
    edge_attr = torch.tensor(ratings, dtype=torch.float)

    print(f"  ✓ Total edges: {edge_index.shape[1]:,}")

    return edge_index, edge_weight, edge_attr


# ============================================================================
# MAIN GRAPH BUILDING FUNCTION
# ============================================================================

def build_graph():
    """Main function to build PyTorch Geometric graph"""

    print("=" * 80)
    print("  AMAZON GRAPH BUILDER")
    print("=" * 80)
    print(f"Embedding model: {EMBEDDING_MODEL} (768 dims)")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Format: .pkl (PyTorch native)")
    print(f"Memory optimization: Streaming embeddings to disk")
    print("=" * 80)

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================

    print("\n" + "=" * 80)
    print("[1/7] LOADING DATA")
    print("=" * 80)

    print("\n  Loading reviews...")
    reviews = pd.read_csv(REVIEWS_FILE)
    print(f"    ✓ Loaded {len(reviews):,} reviews")

    print("\n  Loading metadata...")
    metadata = pd.read_csv(METADATA_FILE)
    print(f"    ✓ Loaded {len(metadata):,} products")

    # Filter to common items
    items_in_reviews = set(reviews['parent_asin'].unique())
    metadata = metadata[metadata['parent_asin'].isin(items_in_reviews)]
    print(f"    ✓ Filtered to {len(metadata):,} products")

    # ========================================================================
    # STEP 2: ENCODE NODE IDs
    # ========================================================================

    print("\n" + "=" * 80)
    print("[2/7] ENCODING NODE IDs")
    print("=" * 80)

    common_items = set(reviews['parent_asin'].unique()) & set(metadata['parent_asin'].unique())
    reviews = reviews[reviews['parent_asin'].isin(common_items)]

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_encoder.fit(reviews['user_id'].unique())
    item_encoder.fit(list(common_items))

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    print(f"\n  ✓ Encoded {num_users:,} users")
    print(f"  ✓ Encoded {num_items:,} items")

    # ========================================================================
    # STEP 3: BUILD USER FEATURES
    # ========================================================================

    print("\n" + "=" * 80)
    print("[3/7] BUILDING USER FEATURES")
    print("=" * 80)

    user_profiles = build_user_profile_features(reviews)
    user_profiles = user_profiles.set_index('user_id').loc[user_encoder.classes_].reset_index()

    user_features = torch.tensor(
        user_profiles[['avg_rating_norm', 'rating_std_norm', 'review_count_norm', 'activity_span_norm']].values,
        dtype=torch.float
    )

    print(f"\n  ✓ User features shape: {user_features.shape}")

    # ========================================================================
    # STEP 4: BUILD ITEM FEATURES
    # ========================================================================

    print("\n" + "=" * 80)
    print("[4/7] BUILDING ITEM FEATURES")
    print("=" * 80)

    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"    ✓ Model loaded")

    metadata = metadata.set_index('parent_asin').loc[item_encoder.classes_].reset_index()

    item_features = build_item_features(metadata, embedding_model)

    # ========================================================================
    # STEP 5: BUILD EDGES
    # ========================================================================

    print("\n" + "=" * 80)
    print("[5/7] BUILDING EDGES")
    print("=" * 80)

    edge_index, edge_weight, edge_attr = build_edges(reviews, user_encoder, item_encoder)

    # ========================================================================
    # STEP 6: CREATE GRAPH
    # ========================================================================

    print("\n" + "=" * 80)
    print("[6/7] CREATING GRAPH")
    print("=" * 80)

    data = HeteroData()

    data['user'].x = user_features
    data['item'].x = item_features

    data['user', 'rates', 'item'].edge_index = edge_index
    data['user', 'rates', 'item'].edge_weight = edge_weight
    data['user', 'rates', 'item'].edge_attr = edge_attr.unsqueeze(1)

    data['item', 'rated_by', 'user'].edge_index = edge_index.flip([0])
    data['item', 'rated_by', 'user'].edge_weight = edge_weight
    data['item', 'rated_by', 'user'].edge_attr = edge_attr.unsqueeze(1)

    data['user'].num_nodes = num_users
    data['item'].num_nodes = num_items

    print("\n  Graph structure:")
    print(data)

    # ========================================================================
    # STEP 7: SAVE
    # ========================================================================

    print("\n" + "=" * 80)
    print("[7/7] SAVING GRAPH")
    print("=" * 80)

    save_dict = {
        'graph': data,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'metadata': {
            'num_users': num_users,
            'num_items': num_items,
            'num_interactions': len(reviews),
            'user_feature_dim': user_features.shape[1],
            'item_feature_dim': item_features.shape[1],
            'embedding_model': EMBEDDING_MODEL
        }
    }

    torch.save(save_dict, OUTPUT_FILE)
    print(f"\n  ✅ Graph saved: {OUTPUT_FILE}")
    print(f"     Format: .pkl (PyTorch native)")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("  GRAPH STATISTICS")
    print("=" * 80)
    print(f"\nNodes:")
    print(f"  Users:  {num_users:,}")
    print(f"  Items:  {num_items:,}")
    print(f"  Total:  {num_users + num_items:,}")
    print(f"\nEdges:")
    print(f"  User -> Item:  {edge_index.shape[1]:,}")
    print(f"  Item -> User:  {edge_index.shape[1]:,}")
    print(f"\nFeatures:")
    print(f"  User dim:  {user_features.shape[1]}")
    print(f"  Item dim:  {item_features.shape[1]}")
    density = edge_index.shape[1] / (num_users * num_items)
    print(f"\nDensity: {density:.8f} ({density * 100:.6f}%)")

    print("\n" + "=" * 80)
    print("  BUILD COMPLETE ✓")
    print("=" * 80)


if __name__ == "__main__":
    build_graph()