import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("src")
from core.constants import PROJECT_ROOT_DIR, os_join


def get_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_metadata():
    src_root = Path(os_join(PROJECT_ROOT_DIR, "src"))
    docs_root = Path(os_join(PROJECT_ROOT_DIR, "docs", "source"))

    if not src_root.exists() or not docs_root.exists():
        print(f"Error: Required directories '{src_root}' or '{docs_root}' not found.")
        return

    python_files = list(src_root.rglob("*.py"))
    print(f"Found {len(python_files)} python files in {src_root}")

    missing_docs = []
    updated_docs = []
    content_sync_needed = []

    for py_path in python_files:
        rel_path = py_path.relative_to(src_root)
        if rel_path.stem == "__init__":
            continue

        doc_filename = f"{rel_path.stem}_{rel_path.suffix.lstrip('.')}.md"
        doc_path = docs_root / rel_path.parent / doc_filename

        if not doc_path.exists():
            missing_docs.append(doc_path)
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Calculate current hash
        current_hash = get_file_hash(py_path)

        # Update lastmod
        mtime = datetime.fromtimestamp(py_path.stat().st_mtime).strftime("%Y-%m-%d")
        new_content, mod_count = re.subn(
            r"(^lastmod:\s*)\d{4}-\d{2}-\d{2}",
            f"\\g<1>{mtime}",
            content,
            flags=re.MULTILINE,
        )

        # Update or add src_hash
        if "src_hash:" in new_content:
            new_content, hash_count = re.subn(
                r"(^src_hash:\s*)[a-fA-F0-9]+",
                f"\\g<1>{current_hash}",
                new_content,
                flags=re.MULTILINE,
            )
            is_different_hash = hash_count > 0 and current_hash not in content
        else:
            # Add src_hash after lastmod if it doesn't exist
            new_content, hash_count = re.subn(
                r"(^lastmod:.*$)",
                f"\\1\nsrc_hash: {current_hash}",
                new_content,
                flags=re.MULTILINE,
            )
            is_different_hash = True

        if new_content != content:
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            status_msg = f"Updated {doc_path} metadata"
            if is_different_hash:
                status_msg += " (Hash Change Detected)"
                content_sync_needed.append((py_path, doc_path))

            print(status_msg)
            updated_docs.append(doc_path)

    print("\n--- Sync Summary ---")
    print(f"Total Python Files: {len(python_files)}")
    print(f"Updated Metadata: {len(updated_docs)}")
    print(f"Missing Docs: {len(missing_docs)}")
    print(f"Content Sync Needed (AI Task): {len(content_sync_needed)}")

    if content_sync_needed:
        print("\n🚀 ACTION REQUIRED: The following files need content synchronization.")
        print("Please follow SOURCE_DOCS_GUIDE.md to express changes into md files:")
        for py_p, doc_p in content_sync_needed:
            print(f"- [ ] {doc_p} (source: {py_p})")

    if missing_docs:
        print("\n⚠️  MISSING DOCUMENTATION:")
        for doc in missing_docs:
            print(f"- {doc}")


if __name__ == "__main__":
    sync_metadata()
