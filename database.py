"""
database.py
-----------
Database connection and session management for the SpamGuard API.

Establishes the SQLAlchemy engine, configures the session factory, and
provides the declarative base class from which all ORM models inherit.
A FastAPI-compatible dependency function is also defined here to supply
a scoped database session to endpoint handlers via dependency injection.

The default storage backend is SQLite, which requires no external server
and is appropriate for development and demonstration deployments. For
production use, the DATABASE_URL environment variable may be set to any
SQLAlchemy-compatible connection string (e.g. PostgreSQL, MySQL).
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# ---------------------------------------------------------------------------
# Connection string resolution
# ---------------------------------------------------------------------------
# The DATABASE_URL environment variable allows the storage backend to be
# swapped without modifying source code. The default targets a local SQLite
# file named spam.db in the application working directory.
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./spam.db')

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------
# The connect_args parameter is required exclusively for SQLite to permit
# concurrent access from multiple threads, which occurs when FastAPI handles
# simultaneous requests. This argument is omitted for other database backends
# that natively support connection pooling and thread safety.
engine = create_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False} if 'sqlite' in DATABASE_URL else {}
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
# SessionLocal is a factory class that produces individual database session
# objects. autocommit=False ensures that changes are not persisted until an
# explicit db.commit() call is made, providing transactional safety.
# autoflush=False prevents SQLAlchemy from issuing premature SQL statements
# before a commit is explicitly requested.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
# All ORM model classes defined in models.py inherit from Base. SQLAlchemy
# uses this shared base to discover and register table definitions when
# Base.metadata.create_all() is invoked at application startup.
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that provides a transactional database session.

    Yields a SQLAlchemy Session object for the duration of a single HTTP
    request. The session is unconditionally closed in the finally block,
    ensuring that database connections are released back to the connection
    pool regardless of whether the request succeeded or raised an exception.

    Yields
    ------
    Session
        An active SQLAlchemy ORM session bound to the configured engine.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
