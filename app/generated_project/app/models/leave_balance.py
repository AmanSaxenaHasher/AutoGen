from app.database import Base
from sqlalchemy import Column, Integer, String, ForeignKey

class LeaveBalance(Base):
    __tablename__ = 'leave_balances'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    balance = Column(Integer)