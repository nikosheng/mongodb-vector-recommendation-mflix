"""
scripts/wait_for_server.py
Polls GET /api/health until the server responds or a timeout is reached.
Exits 0 on success, 1 on timeout.

Usage:
    python3 scripts/wait_for_server.py [url] [timeout_seconds]
"""
import sys
import time
import urllib.request

url     = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/api/health"
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 20
deadline = time.time() + timeout

sys.stdout.write("  Waiting for server")
sys.stdout.flush()

while time.time() < deadline:
    try:
        urllib.request.urlopen(url, timeout=1)
        print(" ready")
        sys.exit(0)
    except Exception:
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.5)

print(" timed out")
sys.exit(1)
