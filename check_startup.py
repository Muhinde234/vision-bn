"""Run this to verify the app starts correctly before using uvicorn."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("Checking imports...")

try:
    from app.config import settings
    print("  [OK] config loaded - APP:", settings.APP_NAME)
    print("  [OK] DB URL:", settings.DATABASE_URL[:30] + "...")
except Exception as e:
    print("  [FAIL] config:", e)
    sys.exit(1)

try:
    from app.main import app
    print("  [OK] FastAPI app loaded")
    routes = [r.path for r in app.routes]
    print("  [OK] Routes registered:", len(routes))
except Exception as e:
    print("  [FAIL] app:", e)
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("All checks passed.")
print("Start the server with:")
print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
print()
print("Then open: http://localhost:8000/docs")
