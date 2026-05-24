"""
Dev helper – creates all database tables directly from SQLAlchemy models.
Run once before starting the server: python create_tables.py
Works with both SQLite (local dev) and PostgreSQL (production).
"""
import asyncio
import sys
import os

# Ensure we run from the backend root
sys.path.insert(0, os.path.dirname(__file__))

async def main():
    from app.db.base import Base
    from app.db.session import engine

    # Import all models so metadata is populated
    import app.models  # noqa: F401

    print("Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("OK - All tables created successfully.")
    print("  You can now run:  uvicorn app.main:app --reload")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
