from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin

pwd_context = CryptContext(schemes=['bcrypt'], default='bcrypt')

def register_user(db: Session, user: UserCreate) -> dict:
    user_dict = user.dict()
    user_dict['password'] = pwd_context.hash(user_dict['password'])
    db_user = db.query(User).filter(User.username == user_dict['username']).first()
    if db_user:
        raise HTTPException(status_code=400, detail='Username already exists')
    db_user = User(**user_dict)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {'message': 'User created successfully'}

def authenticate_user(db: Session, user: UserLogin) -> dict:
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(status_code=400, detail='Invalid username or password')
    access_token = jwt.encode({'sub': db_user.username, 'exp': datetime.utcnow() + timedelta(minutes=30)}, 'secret_key', algorithm='HS256')
    return {'access_token': access_token, 'token_type': 'bearer'}
