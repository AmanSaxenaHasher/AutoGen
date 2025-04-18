from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserLogin
from app.services.auth_service import register_user, authenticate_user
from app.database import get_db

router = APIRouter(
    prefix='/auth',
    tags=['auth']
)

@router.post('/register')
async def register_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    return register_user(db, user)

@router.post('/login')
async def login_endpoint(user: UserLogin, db: Session = Depends(get_db)):
    return authenticate_user(db, user)
