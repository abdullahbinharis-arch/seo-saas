"""
database.py — SQLAlchemy models and session management.

Uses PostgreSQL in production (via DATABASE_URL env var set by Railway).
Falls back to SQLite locally so you can develop without Postgres.
"""

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./seo_audits.db")

# Railway (and some other hosts) expose postgres:// but SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    # SQLite needs this flag; ignored by Postgres
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,   # drop stale connections before use
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id               = Column(String(36), primary_key=True)
    email            = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password  = Column(String, nullable=True)   # null for Google-only accounts
    google_sub       = Column(String(255), nullable=True, index=True)
    created_at       = Column(DateTime, default=datetime.utcnow)


class Audit(Base):
    __tablename__ = "audits"

    id               = Column(String(36), primary_key=True)
    user_id          = Column(String(36), index=True)  # FK to users.id (soft reference)
    keyword          = Column(String(255), nullable=False, index=True)
    target_url       = Column(String(2048), nullable=False)
    location         = Column(String(255))
    status           = Column(String(50), default="completed")
    # Store the full JSON result as text; avoids needing a JSON column type
    # that behaves differently across SQLite and Postgres.
    results_json     = Column(Text, nullable=False)
    api_cost         = Column(Float)
    execution_time   = Column(Float)
    created_at       = Column(DateTime, default=datetime.utcnow, index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables. Safe to call on every startup (no-op if they exist)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a session and ensure it's closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
