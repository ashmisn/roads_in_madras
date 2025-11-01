import sqlalchemy
import os
import sys
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import spacy # Import spaCy

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------------------

try:
    # Use relative import assuming tools.py is in src/
    from . import config
except ImportError:
    # Fallback
    try:
        from src import config
    except ImportError:
        import config

# Use langchain_huggingface wrapper
from langchain_huggingface import HuggingFaceEmbeddings
# Use LangChain's Chroma wrapper
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Load spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded.")
except OSError:
    print("--- ERROR: spaCy model 'en_core_web_sm' not found ---")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None # Set to None so the code can handle its absence

# --- Tool 1: Structured Database Search (No changes) ---
sql_engine = None
try:
    if os.path.exists(config.STRUCTURED_DB_PATH):
        sql_engine = sqlalchemy.create_engine(f"sqlite:///{config.STRUCTURED_DB_PATH}")
        print("Structured DB Engine initialized.")
    else:
        print(f"Error: SQLite DB not found at {config.STRUCTURED_DB_PATH}.")
except Exception as e:
    print(f"Error initializing structured DB engine: {e}")
    sql_engine = None

def structured_search(problem: str = None, category: str = None, type: str = None) -> List[Dict[str, Any]]:
    """Searches the structured SQLite database based on exact filters."""
    # (Keep the existing function body - no changes needed)
    if sql_engine is None:
        print("Error: SQL Engine not available for structured search.")
        return []
    where_clauses = []
    params = {}
    if problem:
        where_clauses.append("problem = :problem")
        params["problem"] = problem
    if category:
        where_clauses.append("category = :category")
        params["category"] = category
    if type:
        where_clauses.append("type LIKE :type") # Use LIKE for flexibility
        params["type"] = f"%{type}%"
    if not where_clauses:
        print("No valid filters provided to structured_search.")
        return []
    select_cols = '`S. No.`, problem, category, type, data, code, clause'
    query_str = f"SELECT {select_cols} FROM {config.SQL_TABLE_NAME} WHERE " + " AND ".join(where_clauses)
    print(f"Executing SQL query: {query_str} with params {params}")
    try:
        with sql_engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query_str), params)
            rows = [dict(row) for row in result.mappings().fetchall()]
            return rows
    except Exception as e:
        print(f"Error executing structured search: {e}")
        return []


# --- Tool 2: Vector Database (Semantic) Search (No changes) ---
langchain_vector_store = None
embedding_function = None
try:
    if os.path.exists(config.VECTOR_STORE_PATH):
        print(f"Initializing embedding function: {config.EMBEDDING_MODEL_NAME}...")
        embedding_function = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding function initialized.")
        print("Loading LangChain Chroma vector store...")
        langchain_vector_store = Chroma(
            persist_directory=config.VECTOR_STORE_PATH,
            embedding_function=embedding_function,
            collection_name=config.VECTOR_COLLECTION_NAME
        )
        print("Verifying vector store connection...")
        _ = langchain_vector_store.similarity_search("test query", k=1)
        print("✅ LangChain Chroma vector store loaded and verified.")
    else:
        print(f"Error: Vector store directory not found at {config.VECTOR_STORE_PATH}. Run data_processor.py.")
except Exception as e:
    print(f"--- ERROR initializing vector DB components ---")
    print(f"Error: {e}")
    langchain_vector_store = None
    embedding_function = None

def semantic_search(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Performs semantic search using LangChain Chroma vector store."""
    # (Keep the existing function body - no changes needed, it uses n_results from agent.py)
    if langchain_vector_store is None:
        print("Error: LangChain Vector Store component is not available.")
        return []
    k_value = n_results # Use the value passed from the agent
    print(f"Executing semantic search for query: '{query}' with k={k_value}")
    try:
        similar_documents: List[Document] = langchain_vector_store.similarity_search(
            query=query,
            k=k_value
        )
        results_metadata = [doc.metadata for doc in similar_documents]
        print(f"--- Retrieved Metadata (Semantic) ---")
        # Ensure metadata is serializable before printing
        serializable_metadata = []
        for meta in results_metadata:
            try:
                json.dumps(meta)
                serializable_metadata.append(meta)
            except TypeError:
                serializable_metadata.append({k: str(v) for k, v in meta.items()})
        print(json.dumps(serializable_metadata, indent=2))
        print(f"------------------------------------")
        return results_metadata # Return original metadata
    except Exception as e:
        print(f"Error during semantic search: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- Tool 3 Helper: Keyword Extraction (REPLACED with spaCy version) ---
def extract_keywords_and_category(query: str) -> Tuple[List[str], Optional[str]]:
    """Extracts keywords (noun chunks, proper nouns) AND tries to identify a category using spaCy."""
    keywords = set() # Use a set to handle duplicates automatically
    category = None
    doc = None

    if nlp is None:
        print("spaCy model not loaded. Falling back to basic keyword extraction.")
        # Insert your previous best regex/stopword fallback logic here if needed
        # For now, just return empty list as spaCy is preferred
        return [], None

    # --- Category Guessing ---
    q_lower = query.lower()
    common_words_to_exclude_category = set()
    if "road sign" in q_lower or " sign" in q_lower:
        category = "Road Sign"
        common_words_to_exclude_category = {'sign', 'signs'}
    elif "road marking" in q_lower or " marking" in q_lower:
        category = "Road Marking"
        common_words_to_exclude_category = {'marking', 'markings'}
    elif "traffic calming" in q_lower or "speed hump" in q_lower or "rumble strip" in q_lower:
        category = "Traffic Calming Measures"
        common_words_to_exclude_category = {'traffic', 'calming', 'measures', 'speed', 'hump', 'rumble', 'strip'}
    print(f"Guessed Category: {category}")

    # --- Keyword Extraction using spaCy ---
    doc = nlp(query)

    # 1. Extract Noun Chunks - often good for multi-word terms
    # Filter out very common/generic noun chunks that aren't specific to road safety items
    # --- FIX: Removed articles ('a', 'an', 'the') from this list ---
    generic_chunks = {
        'road', 'road signs', 'road markings', 'drivers', 'vehicles', 'information',
        'details', 'placement details', 'requirements', 'approach', 'hazard',
        'curve', 'start', 'query', 'context', 'user query', 'area',
        'sign', 'marking', 'location', 'distance', 'section', 'traffic',
        'road users', 'speed', 'design speed', 'km/h', 'guidelines', 'measures',
        'combinations', 'risk', 'issue', 'aspects', 'your query', 'these factors'
    }
    # --- END FIX ---
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        # Remove leading articles for better matching
        articles = {'a ', 'an ', 'the '}
        for article in articles:
            if chunk_text.startswith(article):
                chunk_text = chunk_text[len(article):]
        # Add if not generic and longer than 2 chars
        if chunk_text not in generic_chunks and len(chunk_text) > 2:
            keywords.add(chunk.text.strip()) # Add original casing

    # 2. Extract Proper Nouns (PROPN) - might catch specific names missed by chunks
    for token in doc:
        if token.pos_ == "PROPN" and len(token.text) > 2:
            keywords.add(token.text)

    # 3. Add specific patterns for measurements (like d1, d2, d3, d4) if missed
    measurement_patterns = re.findall(r'\b(d[1-4])\b', query)
    keywords.update(measurement_patterns)

    # 4. Remove keywords that are just the category name itself
    if category:
        category_terms = {term for term in common_words_to_exclude_category if len(term) > 2} # Exclude short terms like 'm'
        keywords = {kw for kw in keywords if kw.lower() not in category_terms}

    # 5. Convert set to list and limit
    final_keywords = list(keywords)[:7] # Limit to 7 keywords

    print(f"Final extracted keywords (spaCy): {final_keywords}")
    return final_keywords, category


# --- Tool 3: Keyword Search ---
# --- FIX: Changed signature to accept keywords list, not query string ---
def keyword_search_sqlite(keywords: List[str], category_hint: Optional[str] = None, n_results: int = 3) -> List[Dict[str, Any]]:
    """Searches SQLite 'type' and 'data' columns for keywords, optionally filtering by category."""
    if sql_engine is None:
        print("Error: SQL Engine not available for keyword search.")
        return []

    # Keywords are now passed directly.
    target_category = category_hint # Use the hint passed from the agent
    # --- END FIX ---

    if not keywords:
        print("No keywords extracted for keyword search.")
        return []

    # Build LIKE clauses
    like_clauses = []
    params = {}
    for i, keyword in enumerate(keywords):
        param_name = f"kw{i}"
        like_clauses.append(f"(LOWER(type) LIKE LOWER(:{param_name}) OR LOWER(data) LIKE LOWER(:{param_name}))")
        params[param_name] = f"%{keyword}%"

    keyword_condition = "(" + " OR ".join(like_clauses) + ")"
    where_conditions = [keyword_condition]

    # Add category filter if identified
    if target_category:
        where_conditions.append("category = :category")
        params["category"] = target_category
        print(f"Applying category filter: {target_category}")

    where_clause_final = " AND ".join(where_conditions)

    select_cols = '`S. No.`, problem, category, type, data, code, clause'
    query_str = f"SELECT {select_cols} FROM {config.SQL_TABLE_NAME} WHERE {where_clause_final} LIMIT :limit"
    params["limit"] = max(n_results, 5) # Fetch at least 5

    print(f"Executing Keyword SQL query: {query_str} with params {params}")
    try:
        with sql_engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query_str), params)
            rows = [dict(row) for row in result.mappings().fetchall()]
            print(f"Keyword search returned {len(rows)} results.")
            return rows
    except Exception as e:
        print(f"Error executing keyword search: {e}")
        return []


# --- Test Functions ---
if __name__ == "__main__":
    print("\n--- Testing Structured Search ---")
    placement_issues = structured_search(problem="Placement Issue")
    print(f"Found {len(placement_issues)} 'Placement Issue' entries via structured search.")
    if placement_issues: print(f"   Example Type: {placement_issues[0].get('type', 'N/A')}")

    print("\n--- Testing Semantic Search ---")
    test_query_semantic = "specifications for stop signs"
    if hasattr(config, 'USE_COMBINED_TEXT_FOR_EMBEDDING') and config.USE_COMBINED_TEXT_FOR_EMBEDDING:
         test_query_semantic = "Type: STOP Sign. Problem: Damaged. Details: The 'STOP' sign..."
    print(f"Testing semantic search with query relevant to embedded text: '{test_query_semantic[:100]}...'")
    sign_info_semantic = semantic_search(query=test_query_semantic, n_results=1)
    if sign_info_semantic:
        print(f"   Top result Type: {sign_info_semantic[0].get('type', 'N/A')}")
        print(f"   Top result S.No.: {sign_info_semantic[0].get('S. No.', 'N/A')}")

    print("\n--- Testing Improved Keyword Extraction & Search (spaCy) ---")
    test_queries_keywords = [
        "What is the meaning and use of a 'No Entry' sign?",
        "For a divided highway section with a design speed of 90 km/h, list the appropriate warning signs and object hazard markers needed when approaching a sharp curve where rockfalls are a known risk. Include details on sign placement relative to the hazard/curve and mention any relevant markings for obstructions, if applicable",
        "Width of rumble strip?",
        "What is distance d4 for transverse bars at 85 km/h?"
    ]
    # --- FIX: Updated test block for new keyword_search_sqlite signature ---
    for test_q in test_queries_keywords:
        print(f"\nTesting keyword search with query: '{test_q}'")
        # Step 1: Extract keywords first (as the agent does)
        extracted_keywords, guessed_category = extract_keywords_and_category(test_q)
        # Step 2: Pass keywords to the search function
        sign_info_keyword = keyword_search_sqlite(
            keywords=extracted_keywords,
            category_hint=guessed_category,
            n_results=2
        )
        print(f"Found {len(sign_info_keyword)} results via keyword search.")
        if sign_info_keyword:
            print(f"   Top result Type: {sign_info_keyword[0].get('type', 'N/A')}")
            print(f"   Top result S.No.: {sign_info_keyword[0].get('S. No.', 'N/A')}")
    # --- END FIX ---