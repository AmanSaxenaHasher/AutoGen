from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
from app.models.leave_request import LeaveRequest
from app.schemas.leave_request import LeaveRequestSchema, LeaveRequestCreate, LeaveRequestUpdate
from app.services.leave_request import (
    create_leave_request,
    get_leave_requests,
    get_leave_request,
    update_leave_request,
    delete_leave_request
)
from app.database import get_db

router = APIRouter(
    prefix='/leave-requests',
    tags=['leave-requests']
)

@router.post('/', response_model=LeaveRequestSchema)
async def create_leave_request_endpoint(leave_request: LeaveRequestCreate, db: Session = Depends(get_db)):
    return create_leave_request(db, leave_request)

@router.get('/', response_model=List[LeaveRequestSchema])
async def get_leave_requests_endpoint(db: Session = Depends(get_db)):
    return get_leave_requests(db)

@router.get('/{leave_request_id}', response_model=LeaveRequestSchema)
async def get_leave_request_endpoint(leave_request_id: int, db: Session = Depends(get_db)):
    return get_leave_request(db, leave_request_id)

@router.put('/{leave_request_id}', response_model=LeaveRequestSchema)
async def update_leave_request_endpoint(leave_request_id: int, leave_request: LeaveRequestUpdate, db: Session = Depends(get_db)):
    return update_leave_request(db, leave_request_id, leave_request)

@router.delete('/{leave_request_id}', response_model=LeaveRequestSchema)
async def delete_leave_request_endpoint(leave_request_id: int, db: Session = Depends(get_db)):
    return delete_leave_request(db, leave_request_id)
