import sys
import os
import logging
import numpy as np 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sentence_transformers import SentenceTransformer
from app.db.database import get_db_connection, get_db_cursor 
from app.core.config import settings 

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ---------------------


def generate_and_store_embeddings():
    """
    Fetches expectations without embeddings, generates them using a
    Sentence Transformer model, and updates the database.
    """
    logger.info(f"Starting embedding generation using model: {MODEL_NAME}")

    # 1. Load the Sentence Transformer model
    try:
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Sentence Transformer model loaded successfully.")
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.exception("Failed to load Sentence Transformer model.")
        return 

    conn = None
    cur = None
    updated_count = 0

    try:
        # 2. Connect to the database
        conn = get_db_connection()
        if not conn:
            logger.error("Could not establish database connection. Exiting.")
            return

        cur = get_db_cursor(conn)
        if not cur:
            logger.error("Could not get database cursor. Exiting.")
            if conn: conn.close()
            return

        # 3. Fetch expectations needing embeddings
        cur.execute("SELECT expectation_id, expectation_text FROM expectations WHERE embedding IS NULL;")
        expectations_to_process = cur.fetchall()

        if not expectations_to_process:
            logger.info("No expectations found needing embedding generation.")
            return

        logger.info(f"Found {len(expectations_to_process)} expectations to process.")

        # 4. Generate and update embeddings
        for record in expectations_to_process:
            exp_id = record['expectation_id']
            exp_text = record['expectation_text']
            logger.debug(f"Processing expectation ID: {exp_id}")

            try:
                embedding_vector = model.encode(exp_text)

                embedding_list = embedding_vector.tolist()

                update_sql = "UPDATE public.expectations SET embedding = %s WHERE expectation_id = %s;"
                cur.execute(update_sql, (embedding_list, exp_id))
                logger.debug(f"Generated and prepared update for expectation ID: {exp_id}")
                updated_count += 1

            except Exception as e:
                logger.error(f"Failed to process or update expectation ID {exp_id}: {e}")

        # 5. Commit all changes if successful
        conn.commit()
        logger.info(f"Successfully generated and stored embeddings for {updated_count} expectations.")

    except Exception as e:
        logger.exception("An error occurred during the embedding generation process.")
        if conn:
            conn.rollback() 
            logger.info("Database changes rolled back due to error.")
    finally:
        # 6. Close cursor and connection
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.info("Database connection closed.")

if __name__ == "__main__":
    generate_and_store_embeddings()