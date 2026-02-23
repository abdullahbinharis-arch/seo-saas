"""
database.py — SQLAlchemy models and session management.

Uses PostgreSQL in production (via DATABASE_URL env var set by Railway).
Falls back to SQLite locally so you can develop without Postgres.
"""

import json
import logging
import os
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger("seo-saas")

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


class Profile(Base):
    __tablename__ = "profiles"

    id                = Column(String(36), primary_key=True)
    user_id           = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    business_name     = Column(String(255), nullable=False)
    website_url       = Column(String(2048), nullable=False)
    business_category = Column(String(255), nullable=True)
    services          = Column(Text, nullable=True)       # JSON array as text
    country           = Column(String(100), nullable=True)
    city              = Column(String(255), nullable=True)
    nap_data          = Column(Text, nullable=True)       # JSON object as text
    is_active         = Column(Boolean, default=True, nullable=False)
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
    # v2 — profile linkage + denormalized fields
    profile_id       = Column(String(36), index=True, nullable=True)
    version          = Column(Integer, nullable=True)
    business_name    = Column(String(255), nullable=True)
    business_type    = Column(String(255), nullable=True)
    pages_crawled    = Column(Integer, nullable=True)


# ---------------------------------------------------------------------------
# Migration helpers — idempotent ALTER TABLE for existing databases
# ---------------------------------------------------------------------------

def _migrate_schema() -> None:
    """Add new columns to existing audits table. Safe to run repeatedly."""
    new_columns = [
        ("profile_id",    "VARCHAR(36)"),
        ("version",       "INTEGER"),
        ("business_name", "VARCHAR(255)"),
        ("business_type", "VARCHAR(255)"),
        ("pages_crawled", "INTEGER"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in new_columns:
            try:
                conn.execute(text(f"ALTER TABLE audits ADD COLUMN {col_name} {col_type}"))
                conn.commit()
                logger.info(f"Migration: added audits.{col_name}")
            except Exception:
                conn.rollback()
                # Column already exists — expected on subsequent startups


def _backfill_profiles() -> None:
    """One-time migration: create Profile records for orphaned audits."""
    db = SessionLocal()
    try:
        # Find audits with no profile_id that belong to a user
        orphaned = (
            db.query(Audit)
            .filter(Audit.profile_id.is_(None), Audit.user_id.isnot(None))
            .order_by(Audit.created_at.asc())
            .all()
        )
        if not orphaned:
            return

        logger.info(f"Backfill: {len(orphaned)} orphaned audits found")

        # Group by (user_id, business_name, target_url)
        groups: dict[tuple, list] = {}
        for audit in orphaned:
            # Try to extract business_name from results_json if not on the column
            bname = audit.business_name
            if not bname and audit.results_json:
                try:
                    data = json.loads(audit.results_json)
                    bname = data.get("business_name", "")
                except Exception:
                    bname = ""
            bname = bname or "Unknown Business"
            key = (audit.user_id, bname, audit.target_url)
            groups.setdefault(key, []).append(audit)

        for (user_id, bname, url), audits in groups.items():
            profile_id = str(uuid.uuid4())
            profile = Profile(
                id=profile_id,
                user_id=user_id,
                business_name=bname,
                website_url=url,
            )
            db.add(profile)
            for i, audit in enumerate(audits, start=1):
                audit.profile_id = profile_id
                audit.version = i
                if not audit.business_name:
                    audit.business_name = bname

        db.commit()
        logger.info(f"Backfill: created {len(groups)} profiles, linked {len(orphaned)} audits")
    except Exception as e:
        db.rollback()
        logger.error(f"Backfill failed: {e}", exc_info=True)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables + run migrations. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
    _migrate_schema()
    _backfill_profiles()


def get_db():
    """Yield a session and ensure it's closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
