from fastapi import APIRouter, HTTPException, status, Depends
from app.db import schemas 
from app.db.database import get_db_connection, get_db_cursor
from typing import List
from app.core import pipeline
import logging
import psycopg2 

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/process_snippet",
    response_model=schemas.ProcessResponse,
    summary="Process text snippet to find and record skill achievements",
    tags=["Evaluation"]
)
async def process_snippet_endpoint(request: schemas.ProcessRequest):
    logger.info(f"Received request to process text snippet: {request.text[:100]}...") 

    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty."
        )

    try:
        achievements_count = await pipeline.process_text_snippet(request.text)
        logger.info(f"Processing complete. Achievements created: {achievements_count}")
        return schemas.ProcessResponse(
            status="Processed",
            achievements_created=achievements_count
        )
    except Exception as e:
        logger.exception("An error occurred during snippet processing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}"
        )


@router.get(
    "/test/employees",
    response_model=List[schemas.Employee], 
    summary="[Test] Get all employees",
    tags=["Testing"] 
)
async def get_all_employees():
    """
    Simple test endpoint to retrieve all employees from the database.
    Verifies database connection and basic SELECT query execution.
    """
    logger.info("Received request for /test/employees")
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection could not be established."
            )

        cur = get_db_cursor(conn)
        if not cur:
             conn.close()
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get database cursor."
            )

        cur.execute("SELECT employee_id, name FROM Employees ORDER BY employee_id;")
        employee_records = cur.fetchall()
        logger.info(f"Retrieved {len(employee_records)} employee records.")

        employees = [schemas.Employee(**record) for record in employee_records]
        return employees

    except psycopg2.Error as db_err: 
        logger.error(f"Database error while fetching employees: {db_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {db_err}"
        )
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching employees.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.debug("Database connection closed for /test/employees request.")
