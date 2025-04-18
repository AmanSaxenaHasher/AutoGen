from pydantic import BaseModel

class LeaveBalanceBase(BaseModel):
    user_id: int
    total_leaves: int
    used_leaves: int
    remaining_leaves: int

class LeaveBalanceCreate(LeaveBalanceBase):
    pass

class LeaveBalanceUpdate(LeaveBalanceBase):
    pass

class LeaveBalanceSchema(LeaveBalanceBase):
    id: int

    class Config:
        orm_mode = True
