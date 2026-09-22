"""Quick start: python run.py [demo|demo-real|bench|calibrate|serve]."""
import sys
sys.path.insert(0, "src")
from flashbulljev.__main__ import main

if __name__ == "__main__":
    main(sys.argv[1:] or ["demo"])
