"""
Startup script for PathSnap FastAPI Backend (Laptop 1).
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn


def main():
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    print(f"Starting PathSnap API Server on {host}:{port}...")
    # Pass app object or module string with app_dir
    uvicorn.run("src.api.server:app", host=host, port=port, app_dir=PROJECT_ROOT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
