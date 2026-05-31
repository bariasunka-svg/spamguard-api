# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Read DB URL from environment variable, default to SQLite file
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./spam.db')

# create_engine: the core SQLAlchemy connection manager
# connect_args only needed for SQLite (handles multi-threading)
engine = create_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False} if 'sqlite' in DATABASE_URL else {}
)

# SessionLocal: a factory that creates new DB session objects
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base: all models (tables) inherit from this
Base = declarative_base()

# Dependency for FastAPI: creates a session, yields it, closes it
def get_db():
    db = SessionLocal()
    try:
        yield db      # the session is available here
    finally:
        db.close()    # ALWAYS close, even if an exception occurred
