from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

global engine
engine = create_engine('postgresql://user:password@host:port/dbname')
Base = declarative_base()
Session = sessionmaker(bind=engine)
def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()