"""
database.py
-----------
MySQL + SQLAlchemy integration for the
Ensemble-Based Hybrid Disaster Prediction System.

Engine
------
  Driver   : PyMySQL  (pure-Python, no C extensions required)
  Dialect  : mysql+pymysql
  Database : disaster_prediction_db  (must be created manually — see below)

  Connection URL is read from the environment variable DATABASE_URL.
  A sane localhost default is provided for local development.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Create the database once (run in MySQL shell or workbench):    │
  │                                                                 │
  │      CREATE DATABASE disaster_prediction_db                     │
  │          CHARACTER SET utf8mb4                                  │
  │          COLLATE utf8mb4_unicode_ci;                            │
  │                                                                 │
  │  Then grant access:                                             │
  │      CREATE USER 'disaster_user'@'localhost'                    │
  │          IDENTIFIED BY 'your_password';                         │
  │      GRANT ALL PRIVILEGES ON disaster_prediction_db.*           │
  │          TO 'disaster_user'@'localhost';                         │
  │      FLUSH PRIVILEGES;                                          │
  └─────────────────────────────────────────────────────────────────┘

Session
-------
  Synchronous SessionLocal — used in routers via get_db() dependency.

Tables
------
  prediction_records — stores every POST /predict request + result.
  Created automatically on startup via Base.metadata.create_all().

Usage
-----
  In routers → inject `db: Session = Depends(get_db)`
  On startup  → call init_db()

Environment variables
---------------------
  DATABASE_URL   Full SQLAlchemy connection URL.
                 Default: mysql+pymysql://root:password@localhost:3306/disaster_prediction_db

  DB_HOST        MySQL host       (default: localhost)
  DB_PORT        MySQL port       (default: 3306)
  DB_USER        MySQL username   (default: root)
  DB_PASSWORD    MySQL password   (default: password)
  DB_NAME        Database name    (default: disaster_prediction_db)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

logger = logging.getLogger("backend.database")

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent

# Load local backend/.env first, then project root .env.
load_dotenv(_BACKEND_DIR / ".env", override=False)
load_dotenv(_BACKEND_DIR.parent / ".env", override=False)

# ---------------------------------------------------------------------------
# Build MySQL connection URL
# ---------------------------------------------------------------------------
#
# Priority:
#   1. Full DATABASE_URL environment variable (highest priority — production)
#   2. Individual DB_* variables  (CI / container deployments)
#   3. Localhost defaults          (local development)
#
# Example production value:
#   DATABASE_URL=mysql+pymysql://disaster_user:s3cr3t@db-server:3306/disaster_prediction_db

_DB_BACKEND: str = os.getenv("DB_BACKEND", "sqlite").strip().lower()

_DB_HOST: str = os.getenv("DB_HOST", "localhost")
_DB_PORT: str = os.getenv("DB_PORT", "3306")
_DB_USER: str = os.getenv("DB_USER", "root")
_DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")
_DB_NAME: str = os.getenv("DB_NAME", "disaster_prediction_db")
_DEFAULT_SQLITE_PATH = "/data/disaster_prediction.db" if Path("/data").exists() else str(_BACKEND_DIR / "disaster_prediction.db")
_SQLITE_PATH: str = os.getenv("SQLITE_PATH", _DEFAULT_SQLITE_PATH)

if os.getenv("DATABASE_URL"):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
elif _DB_BACKEND == "mysql":
    DATABASE_URL = (
        f"mysql+pymysql://{_DB_USER}:{_DB_PASSWORD}"
        f"@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"
        f"?charset=utf8mb4"
    )
else:
    sqlite_path_obj = Path(_SQLITE_PATH)
    if not sqlite_path_obj.is_absolute():
        sqlite_path_obj = (_BACKEND_DIR.parent / sqlite_path_obj).resolve()
    sqlite_path = sqlite_path_obj.as_posix()
    DATABASE_URL = f"sqlite:///{sqlite_path}"

try:
    _url = make_url(DATABASE_URL)
    _SAFE_URL = DATABASE_URL.replace(_url.password or "", "****") if _url.password else DATABASE_URL
except Exception:
    _SAFE_URL = DATABASE_URL

logger.info("Database connection target: %s", _SAFE_URL)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
#
# pool_pre_ping=True  — test each connection before handing it to a request;
#                       silently replaces stale / dropped connections.
# pool_recycle=3600   — recycle connections after 1 h to avoid MySQL
#                       'wait_timeout' disconnects.
# pool_size / max_overflow — tune for expected concurrency.

_engine_kwargs: dict = {
    "echo": False,
}
if DATABASE_URL.startswith("mysql"):
    _engine_kwargs.update(
        {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_size": 5,
            "max_overflow": 10,
        }
    )
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs.update({"connect_args": {"check_same_thread": False}})

engine = create_engine(DATABASE_URL, **_engine_kwargs)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# Declarative base — imported by models.py
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass

# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables defined via Base subclasses if they do not exist.
    Called once on API startup from main.py lifespan.

    Safe to call multiple times — SQLAlchemy uses CREATE TABLE IF NOT EXISTS
    semantics (checkfirst=True is the default).

    Raises
    ------
    sqlalchemy.exc.OperationalError
        If MySQL is unreachable or the database does not exist.
        The caller (main.py lifespan) catches this and logs a warning.
    """
    from models import PredictionRecord  # noqa: F401 — registers the ORM mapping

    Base.metadata.create_all(bind=engine)
    logger.info(
        "Database initialised. Tables created / verified. URL: %s",
        _SAFE_URL,
    )


def is_db_connected() -> bool:
    """
    Return True if the MySQL engine can execute a trivial query.

    Used by:
      - GET /health    (db_connected field)
      - _db_guard()    (history_router)
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.debug("MySQL connectivity check failed: %s", exc)
        return False


def get_db_engine_label() -> str:
    """Return a user-facing label for the configured database backend."""
    return "SQLite" if DATABASE_URL.startswith("sqlite") else "MySQL"


# ---------------------------------------------------------------------------
# FastAPI dependency — inject a database session per request
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy Session bound to MySQL and guarantee it is closed
    after the request, even if an exception is raised mid-handler.

    Usage in a router
    -----------------
        from fastapi import Depends
        from sqlalchemy.orm import Session
        from database import get_db

        @router.post("/example")
        def example(db: Session = Depends(get_db)):
            records = db.query(SomeModel).all()
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
