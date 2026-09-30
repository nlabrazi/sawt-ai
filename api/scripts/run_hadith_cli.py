"""Prepare a persistent terminal environment inside Docker, separate from the API."""

import os
from pathlib import Path
import subprocess
import sys

API_DIR = Path(__file__).resolve().parents[1]


def main():
    environment = API_DIR / ".cache" / "hadith-cli-venv"
    python = environment / "bin" / "python"
    check = "import pydantic, sentence_transformers, transformers; assert sentence_transformers.__version__ == '3.4.1'; assert transformers.__version__ == '4.48.3'"
    if not python.is_file() or subprocess.run([str(python), "-c", check], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        print("Première utilisation : préparation des dépendances de recherche dans Docker…", flush=True)
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True)
        requirements = [line for line in (API_DIR / "requirements.txt").read_text().splitlines()
                        if line.startswith(("sentence-transformers==", "transformers=="))]
        subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", *requirements], check=True)
    os.execv(str(python), [str(python), str(API_DIR / "scripts" / "search_hadith.py"), "--variant", "benchmark", *sys.argv[1:]])


if __name__ == "__main__":
    main()
