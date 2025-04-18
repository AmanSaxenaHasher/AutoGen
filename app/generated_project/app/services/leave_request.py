from typing import List
from sqlalchemy.orm import Session
from app.models.leave_request import LeaveRequest
from app.schemas.leave_request import LeaveRequestCreate, LeaveRequestUpdate

def create_leave_request(db: Session, leave_request: LeaveRequestCreate) -> LeaveRequest:
    new_leave_request = LeaveRequest(**leave_request.dict())
    db.add(new_leave_request)
    db.commit()
    db.refresh(new_leave_request)
    return new_leave_request

def get_leave_requests(db: Session) -> List[LeaveRequest]:
    return db.query(LeaveRequest).all()

def get_leave_request(db: Session, leave_request_id: int) -> LeaveRequest:
    leave_request = db.query(LeaveRequest).filter(LeaveRequest.id == leave_request_id).first()
    if not leave_request:
        raise HTTPException(status_code=404, detail="Leave request not found")
    return leave_request

def update_leave_request(db: Session, leave_request_id: int, leave_request: LeaveRequestUpdate) -> LeaveRequest:
    existing_leave_request = db.query(LeaveRequest).filter(LeaveRequest.id == leave_request_id).first()
    if not existing_leave_request:
        raise HTTPException(status_code=404, detail="Leave request not found")
    for key, value in leave_request.dict().items():
        setattr(existing_leave_request, key, value)
    db.commit()
    db.refresh(existing_leave_request)
    return existing_leave_request

def delete_leave_request(db: Session, leave_request_id: int) -> LeaveRequest:
    leave_request = db.query(LeaveRequest).filter(LeaveRequest.id == leave_request_id).first()
    if not leave_request:
        raise HTTPException(status_code=404, detail="Leave request not found")
    db.delete(leave_request)
    db.commit()
    return leave_request
