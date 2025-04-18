from fastapi import FastAPI, Depends
from app.api.routes import leave_requests, auth
from app.database import engine
from app.models import Base

app = FastAPI(
    title='Employee Leave Management System',
    description='API for Employee Leave Management System',
    version='1.0.0'
)

app.include_router(leave_requests.router)
app.include_router(auth.router)

Base.metadata.create_all(engine)