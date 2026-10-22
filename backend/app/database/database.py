"""
Database engine / session management.

SQLite is used for development (see config.DATABASE_URL). Nothing here is
SQLite-specific except the `connect_args` check_same_thread flag, which is a
no-op for Postgres — so switching DATABASE_URL to a postgresql+psycopg2://
URL later requires no code changes.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    # Import models so they register on Base.metadata before create_all.
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
