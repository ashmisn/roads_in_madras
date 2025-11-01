import pandas as pd
import sqlalchemy
import os
import sys
import numpy as np # Needed for embedding type checking


# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------------------

try:
    from src import config
except ImportError:
    import config

# Use sentence-transformers directly for embedding here
from sentence_transformers import SentenceTransformer
# Use Chroma's client directly for adding pre-computed embeddings
import chromadb
from chromadb.utils import embedding_functions # To handle embedding function within Chroma client if needed

# Problem cleaning map remains the same
PROBLEM_CLEANING_MAP = {
    "Wrongly Placed": "Placement Issue",
    "Improper Placement": "Placement Issue",
    "Non-Retro Reflective": "Non-Retroreflective"
}

def load_and_clean_data(csv_path):
    """Loads the raw CSV and standardizes the 'problem' column."""
    print(f"Loading raw data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        # Ensure consistent data types, especially for text fields
        for col in ['problem', 'category', 'type', 'data', 'code', 'clause']:
             # Use .loc to avoid SettingWithCopyWarning if df is a slice
             df.loc[:, col] = df[col].astype(str).fillna('') # Convert to string, fill NaN with empty string
    except FileNotFoundError:
        print(f"--- ERROR: File not found: {csv_path} ---")
        return None
    except Exception as e:
        print(f"--- ERROR loading data: {e} ---")
        return None

    print(f"Loaded {len(df)} raw entries.")
    # Use .loc to modify DataFrame slice safely
    df.loc[:, 'problem'] = df['problem'].replace(PROBLEM_CLEANING_MAP)
    print("Standardized 'problem' column.")
    print("\n--- Cleaned 'problem' value counts ---")
    print(df['problem'].value_counts())
    print("--------------------------------------\n")
    return df

def create_structured_db(df):
    
    """Creates a SQLite database for fast, structured filtering."""
    print("Creating structured SQLite database...")
    try:
        # Ensure directory exists before creating engine
        db_dir = os.path.dirname(config.STRUCTURED_DB_PATH)
        if not os.path.exists(db_dir):
             os.makedirs(db_dir)
             print(f"Created directory: {db_dir}")

        engine = sqlalchemy.create_engine(f"sqlite:///{config.STRUCTURED_DB_PATH}")
        df.to_sql(
            config.SQL_TABLE_NAME,
            engine,
            if_exists='replace',
            index=False,
            dtype={ # Explicit types
                "S. No.": sqlalchemy.types.Integer,
                "problem": sqlalchemy.types.String,
                "category": sqlalchemy.types.String,
                "type": sqlalchemy.types.String,
                "data": sqlalchemy.types.Text, # Use Text for potentially long data
                "code": sqlalchemy.types.String,
                "clause": sqlalchemy.types.String
            }
        )
        print(f"✅ Success: Structured DB created at {config.STRUCTURED_DB_PATH}")
    except Exception as e:
        print(f"--- ERROR creating structured DB: {e} ---")

def create_vector_db(df):
    """Creates/updates a Chroma vector database using SentenceTransformer."""
    print(f"Creating/Updating vector database using model: {config.EMBEDDING_MODEL_NAME}...")

    try:
        print("Loading embedding model...")
        # Use SentenceTransformer directly for encoding on CPU
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device='cpu')
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"--- ERROR loading embedding model: {e} ---")
        return

    texts_to_embed = []
    metadatas = []

    if config.USE_COMBINED_TEXT_FOR_EMBEDDING:
        print("Using combined 'problem. category. type.' text for embeddings.")
        for index, row in df.iterrows():
            # Ensure values are strings before joining
            problem = str(row.get('problem', ''))
            category = str(row.get('category', ''))
            type_val = str(row.get('type', ''))
            combined_text = f"Problem: {problem}. Category: {category}. Type: {type_val}."
            texts_to_embed.append(combined_text)
            # Metadata includes the full original row data
            metadatas.append(row.to_dict())
    else:
        print("Using 'data' column text for embeddings.")
        texts_to_embed = df['data'].astype(str).fillna('').tolist() # Ensure strings
        # Metadata should still include the 'data' field for consistency
        metadatas = df.to_dict('records') # Full row data as metadata

    if not texts_to_embed:
        print("Error: No texts generated for embedding.")
        return

    print(f"Preparing to embed {len(texts_to_embed)} documents...")
    try:
        # Embed texts - SentenceTransformer handles batching internally
        all_embeddings = model.encode(texts_to_embed, show_progress_bar=True)

        # Check if embeddings are numpy arrays and convert to list if necessary
        if isinstance(all_embeddings, np.ndarray):
            all_embeddings = all_embeddings.tolist()

        print(f"Generated {len(all_embeddings)} embeddings.")

        if len(all_embeddings) != len(texts_to_embed):
            print(f"Error: Mismatch between texts ({len(texts_to_embed)}) and embeddings ({len(all_embeddings)}).")
            return

    except Exception as e:
        print(f"--- ERROR during text embedding: {e} ---")
        return

    print("Connecting to Chroma database...")
    try:
        # Ensure directory exists before creating client
        db_dir = config.VECTOR_STORE_PATH
        if not os.path.exists(db_dir):
             os.makedirs(db_dir)
             print(f"Created directory: {db_dir}")

        # Use Chroma's PersistentClient for saving data
        client = chromadb.PersistentClient(path=config.VECTOR_STORE_PATH)

        # Get or create the collection.
        # Use a default embedding function for potential future queries directly via client,
        # but we are adding embeddings manually here.
        # ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME) # Optional
        collection = client.get_or_create_collection(
             name=config.VECTOR_COLLECTION_NAME,
             # embedding_function=ef # Optional: Set default EF for the collection
             metadata={"hnsw:space": "cosine"} # Explicitly set cosine distance for BGE models
        )

        # Prepare data for Chroma: IDs must be strings
        ids = [str(i) for i in range(len(texts_to_embed))] # Simple sequential IDs as strings

        print(f"Adding/Updating {len(ids)} documents in collection '{config.VECTOR_COLLECTION_NAME}'...")
        # Use upsert to add new or update existing documents by ID
        # Batching upserts might be more efficient for very large datasets
        batch_size = 500 # Example batch size
        for i in range(0, len(ids), batch_size):
            print(f"Upserting batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
            collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=all_embeddings[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                documents=texts_to_embed[i:i+batch_size] # Store the text that was actually embedded
            )

        print(f"✅ Success: Vector DB collection '{config.VECTOR_COLLECTION_NAME}' updated/created at {config.VECTOR_STORE_PATH}")

    except Exception as e:
        print(f"--- ERROR interacting with Chroma database: {e} ---")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the entire data processing pipeline."""
    df = load_and_clean_data(config.RAW_DATA_PATH)
    if df is not None:
        processed_dir = os.path.dirname(config.STRUCTURED_DB_PATH) # Same dir for both DBs
        if not os.path.exists(processed_dir):
            try:
                os.makedirs(processed_dir)
                print(f"Created directory: {processed_dir}")
            except Exception as e:
                 print(f"--- ERROR creating directory {processed_dir}: {e} ---")
                 return # Stop if cannot create directory

        create_structured_db(df)
        create_vector_db(df)
        print("\n🎉 All data processing complete!")

if __name__ == "__main__":
    main()