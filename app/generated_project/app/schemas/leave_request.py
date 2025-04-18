from pydantic import BaseModel
from datetime import date

class LeaveRequestSchema(BaseModel):
    id: int
    user_id: int
    start_date: date
    end_date: date
    type: str
    description: str
    status: str

class LeaveRequestCreate(BaseModel):
    start_date: date
    end_date: date
    type: str
    description: str

class LeaveRequestUpdate(BaseModel):
    start_date: date
    end_date: date
    type: str
    description: str