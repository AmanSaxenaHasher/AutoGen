from pydantic import BaseModel

class UserSchema(BaseModel):
    id: int
    username: str
    password: str
    role: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str