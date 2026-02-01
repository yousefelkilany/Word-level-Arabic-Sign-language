import os
import shutil
import subprocess
import sys

sys.path.append("src")
from core.constants import PROJECT_ROOT_DIR, os_join

# Configuration
SYNC_FILES = ["quartz.config.ts", "quartz.layout.ts"]
SOURCE_DIR = os_join(os.path.dirname(PROJECT_ROOT_DIR), "quartz-fork")
DEST_DIR = os_join(PROJECT_ROOT_DIR, "quartz-config")


def sync(sync_file):
    SOURCE_FILE = os_join(SOURCE_DIR, sync_file)
    DEST_FILE = os_join(DEST_DIR, sync_file)
    print(f"Syncing {SOURCE_FILE} to {DEST_FILE}...")

    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file {SOURCE_FILE} not found.")
        sys.exit(1)

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    try:
        shutil.copy2(SOURCE_FILE, DEST_FILE)
        print("File copied successfully.")

        # Stage the file in git
        subprocess.run(["git", "add", DEST_FILE], check=True, cwd=PROJECT_ROOT_DIR)
        print(f"Staged {DEST_FILE} in Git.")

    except Exception as e:
        print(f"Error during sync: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Running pre-commit config sync...")

    for sync_file in SYNC_FILES:
        sync(sync_file)
