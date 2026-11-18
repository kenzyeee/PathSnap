"""
Startup script for PathSnap Evaluator Streamlit UI (Laptop 2).
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    port = os.getenv("UI_PORT", "8501")
    print(f"Starting PathSnap Streamlit UI on port {port}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        os.path.join(PROJECT_ROOT, "src", "ui", "app.py"),
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ]
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
