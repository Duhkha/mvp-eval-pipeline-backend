# app/db/schemas.py
from pydantic import BaseModel

class ProcessRequest(BaseModel):
    text: str

class ProcessResponse(BaseModel):
    status: str
    achievements_created: int = 0
    message: str | None = None

# --- Add this new model ---
class Employee(BaseModel):
    employee_id: int
    name: str

    # Optional: Allow creating this model directly from ORM objects or dicts
    # Requires pydantic >= 2.0
    # Remove if using older pydantic or find it unnecessary
    class Config:
        from_attributes = True # Updated for Pydantic v2 (was orm_mode = True)
# --- End of new model ---