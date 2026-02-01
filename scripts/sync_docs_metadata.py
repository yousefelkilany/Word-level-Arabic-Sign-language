import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("src")
from core.constants import PROJECT_ROOT_DIR, os_join


def sync_metadata():
    src_root = Path(os_join(PROJECT_ROOT_DIR, "src"))
    docs_root = Path(os_join(PROJECT_ROOT_DIR, "docs", "source"))

    if not src_root.exists() or not docs_root.exists():
        print(f"Error: Required directories '{src_root}' or '{docs_root}' not found.")
        return

    python_files = list(src_root.rglob("*.py"))
    print(f"Found {len(python_files)} python files in {src_root}")

    updated_count = 0
    missing_docs = []

    for py_path in python_files:
        # Mapping logic: src/path/file.py -> docs/source/path/file_py.md
        rel_path = py_path.relative_to(src_root)
        if rel_path.stem == "__init__":
            continue

        # Mirror the folder structure and naming convention
        # example: src/modelling/dashboard/views.py -> docs/source/modelling/dashboard/views_py.md
        doc_filename = f"{rel_path.stem}_{rel_path.suffix.lstrip('.')}.md"
        doc_path = docs_root / rel_path.parent / doc_filename

        if not doc_path.exists():
            missing_docs.append(doc_path)
            continue

        # Get Python file modification date from OS
        mtime = datetime.fromtimestamp(py_path.stat().st_mtime).strftime("%Y-%m-%d")

        # Read documentation file
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Update lastmod in frontmatter
        # Pattern matches 'lastmod: YYYY-MM-DD' within YAML frontmatter
        new_content, count = re.subn(
            r"(^lastmod:\s*)\d{4}-\d{2}-\d{2}",
            f"\\g<1>{mtime}",
            content,
            flags=re.MULTILINE,
        )

        if count > 0 and new_content != content:
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated {doc_path} lastmod to {mtime}")
            updated_count += 1

    print(f"Sync complete. Updated: {updated_count}, Missing Docs: {len(missing_docs)}")

    if missing_docs:
        print(f"List of {len(missing_docs)} missing docs")
        for doc in missing_docs:
            print(doc)


if __name__ == "__main__":
    sync_metadata()
