#!/usr/bin/env python3
"""Run the validated V28 Generic-ID peer-consensus stage."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / 'scripts_v28' / '01_generic_unknown_peer_consensus.py')], cwd=ROOT, check=True)