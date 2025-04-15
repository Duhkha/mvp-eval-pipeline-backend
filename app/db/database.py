# app/db/database.py
import psycopg2
import psycopg2.extras # For dictionary cursor
from app.core.config import settings # Import the settings instance
import logging # Use logging instead of print for messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes a connection to the PostgreSQL database using settings."""
    conn = None # Initialize conn to None
    try:
        conn = psycopg2.connect(
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            host=settings.DB_HOST,
            port=settings.DB_PORT
        )
        logger.info("Database connection successful")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to database: {e}")
        # If connection fails, ensure conn is None before returning
        # (it might be partially initialized in some error cases)
        if conn:
             try:
                 conn.close() # Attempt to close if partially open
             except Exception as close_err:
                 logger.error(f"Error closing partially opened connection: {close_err}")
        return None

def get_db_cursor(conn):
    """Gets a dictionary cursor from a connection."""
    if conn is None:
        logger.error("Cannot get cursor from None connection.")
        return None
    # Return rows as dictionaries
    return conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# Example usage pattern (will be used in endpoints)
# conn = None
# cur = None
# try:
#     conn = get_db_connection()
#     if conn:
#         cur = get_db_cursor(conn)
#         # cur.execute(...)
#         # results = cur.fetchall()
#         # conn.commit() # If modifying data
# except Exception as e:
#     logger.error(f"Database operation failed: {e}")
#     if conn: conn.rollback() # Rollback on error if modifying data
# finally:
#     if cur: cur.close()
#     if conn: conn.close()