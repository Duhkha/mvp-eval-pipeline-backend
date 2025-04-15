import logging
import nltk 
from datetime import date
import psycopg2 

from app.models import loader 
from app.db.database import get_db_connection, get_db_cursor 
from app.core.config import settings 

logger = logging.getLogger(__name__)

def segment_sentences(text: str) -> list[str]:
    """Splits text into sentences using NLTK."""
    try:
        return nltk.sent_tokenize(text)
    except Exception as e:
        logger.error(f"NLTK sentence tokenization failed: {e}")
        logger.warning("Falling back to splitting by newline for sentence segmentation.")
        return text.splitlines()

def find_employee_in_sentence(sentence: str, ner_model) -> tuple[str | None, int | None]:
    """
    Uses the NER model to find exactly one PERSON entity in a sentence
    and looks up their ID in the Employees table.

    Returns:
        tuple[str | None, int | None]: (employee_name, employee_id) if found, else (None, None)
    """
    if not ner_model:
        logger.error("NER model is not loaded. Cannot find employee.")
        return None, None

    try:
        ner_results = ner_model(sentence)
        person_entities = [entity for entity in ner_results if entity['entity_group'] == 'PER']

        if len(person_entities) == 1:
            employee_name = person_entities[0]['word']
            logger.debug(f"Found potential employee: '{employee_name}' in sentence: '{sentence}'")

            conn = None
            cur = None
            try:
                conn = get_db_connection()
                if conn:
                    cur = get_db_cursor(conn)
                    cur.execute("SELECT employee_id FROM Employees WHERE name = %s", (employee_name,))
                    result = cur.fetchone()
                    if result:
                        employee_id = result['employee_id']
                        logger.info(f"Matched NER name '{employee_name}' to Employee ID: {employee_id}")
                        return employee_name, employee_id
                    else:
                        logger.info(f"NER name '{employee_name}' not found in Employees table.")
                        return None, None 
                else:
                    logger.error("DB connection failed during employee lookup.")
                    return None, None
            except psycopg2.Error as db_err:
                logger.error(f"Database error looking up employee '{employee_name}': {db_err}")
                return None, None
            finally:
                if cur: cur.close()
                if conn: conn.close()
        elif len(person_entities) > 1:
            logger.debug(f"Skipping sentence due to multiple PERSON entities: {person_entities}")
            return None, None
        else:
            return None, None

    except Exception as e:
        logger.exception(f"Error during NER processing or employee lookup for sentence: '{sentence}'")
        return None, None

def find_best_expectation_match(sentence: str, sentence_model) -> tuple[int | None, float]:
    """
    Generates an embedding for the sentence and finds the closest
    expectation in the DB using pgvector cosine distance (<=>).

    Returns:
        tuple[int | None, float]: (expectation_id, distance) of the best match,
                                 or (None, float('inf')) if no match or error.
    """
    if not sentence_model:
        logger.error("Sentence Transformer model is not loaded. Cannot find expectation match.")
        return None, float('inf') 

    logger.debug(f"Generating embedding for sentence: '{sentence[:50]}...'")
    try:
        # 1. Generate sentence embedding
        embedding_vector = sentence_model.encode(sentence)
        embedding_list = embedding_vector.tolist()
        logger.debug("Sentence embedding generated successfully.")
    except Exception as e:
        logger.exception(f"Error generating sentence embedding: {e}")
        return None, float('inf') 

    # 2. Query database for the closest expectation
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("DB connection failed during expectation matching.")
            return None, float('inf')

        cur = get_db_cursor(conn)
        if not cur:
             logger.error("Failed to get database cursor during expectation matching.")
             if conn: conn.close()
             return None, float('inf')

        sql = """
            SELECT expectation_id, embedding <=> %s::vector AS distance
            FROM Expectations
            ORDER BY distance ASC
            LIMIT 1;
        """
        cur.execute(sql, (embedding_list,)) 
        result = cur.fetchone()

        if result:
            expectation_id = result['expectation_id']
            distance = result['distance']
            logger.debug(f"Closest expectation found: ID {expectation_id}, Distance: {distance:.4f}")
            return expectation_id, distance
        else:
            logger.warning("No expectations found in the database to compare against.")
            return None, float('inf') 

    except psycopg2.Error as db_err:
        logger.error(f"Database error during expectation matching: {db_err}")
        return None, float('inf') 
    except Exception as e:
        logger.exception(f"Unexpected error during expectation matching: {e}")
        return None, float('inf')
    finally:
        if cur: cur.close()
        if conn: conn.close()
        logger.debug("DB connection closed for expectation matching.")


def record_achievement(employee_id: int, expectation_id: int, sentence: str) -> bool:
    """
    Inserts a new record into the EmployeeAchievements table.

    Returns:
        True if insertion was successful, False otherwise.
    """
    logger.info(f"Attempting to record achievement: EmpID={employee_id}, ExpID={expectation_id}, Sentence='{sentence[:50]}...'")
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("DB connection failed. Cannot record achievement.")
            return False

        cur = get_db_cursor(conn)
        if not cur:
             logger.error("Failed to get database cursor. Cannot record achievement.")
             if conn: conn.close()
             return False

        sql = """
            INSERT INTO EmployeeAchievements
                (employee_id, expectation_id, date_achieved, evidence_snippet)
            VALUES
                (%s, %s, CURRENT_DATE, %s);
        """
        cur.execute(sql, (employee_id, expectation_id, sentence))

        conn.commit()
        logger.info("Achievement recorded successfully in the database.")
        return True

    except psycopg2.Error as db_err:
        logger.error(f"Database error recording achievement: {db_err}")
        if conn:
            conn.rollback() 
            logger.info("Database transaction rolled back.")
        return False 
    except Exception as e:
        logger.exception(f"Unexpected error recording achievement: {e}")
        if conn:
            conn.rollback() 
            logger.info("Database transaction rolled back.")
        return False 
    finally:
        if cur: cur.close()
        if conn: conn.close()
        logger.debug("DB connection closed for achievement recording.")


# --- Main Pipeline Function ---
async def process_text_snippet(text: str) -> int:
    """
    Main pipeline function to process text, find employees, match skills (partially implemented),
    and record achievements (placeholder).

    Returns:
        int: The number of achievements successfully recorded (currently always 0).
    """
    logger.info(f"Starting pipeline processing for text: {text[:100]}...")
    achievements_created = 0 

    # 1. Check if models are loaded 
    ner_model = loader.ner_model_instance
    sentence_model = loader.sentence_transformer_instance
    if not ner_model or not sentence_model:
        logger.error("ML models not loaded properly. Aborting pipeline.")
        return 0

    # 2. Segment text into sentences
    sentences = segment_sentences(text)
    logger.info(f"Segmented text into {len(sentences)} sentences.")

    # 3. Process each sentence
    for sentence in sentences:
        if not sentence.strip(): 
            continue

        logger.debug(f"Processing sentence: '{sentence}'")

        # 3a. Identify Employee (NER) + Match in DB
        employee_name, employee_id = find_employee_in_sentence(sentence, ner_model)

        if employee_id and employee_name: 
            # 3b. Semantic Matching (Generate embedding + DB Query)
            expectation_id, distance = find_best_expectation_match(sentence, sentence_model)

            if expectation_id is not None:
                logger.info(f"Found best match: Expectation ID {expectation_id} with distance {distance:.4f}")

                # 3c. Record Achievement (Threshold Check + DB Insert)
                if distance < settings.SIMILARITY_THRESHOLD:
                    logger.info(f"Match distance ({distance:.4f}) is below threshold ({settings.SIMILARITY_THRESHOLD}). Attempting to record achievement.")

                    success = record_achievement(employee_id, expectation_id, sentence)
                    if success:
                       achievements_created += 1 
                       logger.info(f"Achievement recorded successfully (Total recorded in this request: {achievements_created}).")
                    else:
                       logger.error(f"Failed to record achievement for EmpID={employee_id}, ExpID={expectation_id}.")

                else:
                    logger.info(f"Match distance ({distance:.4f}) is above threshold ({settings.SIMILARITY_THRESHOLD}). No achievement recorded.")
            else:
                logger.info("No matching expectation found for this sentence.")
        else:
            logger.debug("Sentence skipped (no single known employee found).")

    logger.info(f"Pipeline processing finished. Total achievements recorded in this request: {achievements_created}")
    return achievements_created 
