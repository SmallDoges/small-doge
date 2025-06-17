# Copyright 2025 The SmallDoge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Database configuration and management for SmallDoge WebUI
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from small_doge.webui.backend.smalldoge_webui.env import DATABASE_URL, ENV
from small_doge.webui.backend.smalldoge_webui.constants import DB_CONFIG

log = logging.getLogger(__name__)

# Database engine configuration
engine_args = {
    "echo": ENV == "dev",
    "pool_pre_ping": True,
}

# SQLite specific configuration
if DATABASE_URL.startswith("sqlite"):
    engine_args.update({
        "poolclass": StaticPool,
        "connect_args": {
            "check_same_thread": False,
            "timeout": 30,
        },
    })
else:
    # PostgreSQL/MySQL configuration
    engine_args.update({
        "pool_size": DB_CONFIG.POOL_SIZE,
        "max_overflow": DB_CONFIG.MAX_OVERFLOW,
        "pool_timeout": DB_CONFIG.POOL_TIMEOUT,
        "pool_recycle": DB_CONFIG.POOL_RECYCLE,
    })

# Create database engine
engine = create_engine(DATABASE_URL, **engine_args)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

# Metadata for table operations
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        log.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Get database session with context manager.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        log.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        log.info("Database tables created successfully")
    except Exception as e:
        log.error(f"Failed to create database tables: {e}")
        raise


def drop_tables():
    """Drop all database tables."""
    try:
        Base.metadata.drop_all(bind=engine)
        log.info("Database tables dropped successfully")
    except Exception as e:
        log.error(f"Failed to drop database tables: {e}")
        raise


def reset_database():
    """Reset database by dropping and recreating all tables."""
    log.warning("Resetting database - all data will be lost!")
    drop_tables()
    create_tables()


def check_database_connection():
    """Check if database connection is working."""
    try:
        from sqlalchemy import text
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        log.info("Database connection successful")
        return True
    except Exception as e:
        log.error(f"Database connection failed: {e}")
        return False


def get_database_info():
    """Get database information."""
    return {
        "url": DATABASE_URL,
        "engine": str(engine.url),
        "pool_size": getattr(engine.pool, "size", None),
        "checked_out": getattr(engine.pool, "checkedout", None),
        "overflow": getattr(engine.pool, "overflow", None),
    }


# Initialize database on import
if __name__ != "__main__":
    try:
        # Check connection
        if check_database_connection():
            # Create tables if they don't exist
            create_tables()
        else:
            log.error("Failed to establish database connection")
    except Exception as e:
        log.error(f"Database initialization failed: {e}")
        raise
