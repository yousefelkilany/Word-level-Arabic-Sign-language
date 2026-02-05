import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("src")
from core.constants import PROJECT_ROOT_DIR


def get_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_metadata():
    sync_roots = [
        (
            Path(PROJECT_ROOT_DIR) / "src",
            Path(PROJECT_ROOT_DIR) / "docs" / "source",
            "API/Source",
            ["*.py"],
        ),
        (
            Path(PROJECT_ROOT_DIR) / "scripts",
            Path(PROJECT_ROOT_DIR) / "docs" / "source" / "scripts",
            "Scripts",
            ["*.py"],
        ),
        (
            Path(PROJECT_ROOT_DIR) / "static",
            Path(PROJECT_ROOT_DIR) / "docs" / "source" / "frontend",
            "Frontend",
            ["*.js", "*.html"],
        ),
    ]

    total_source_files = 0
    total_updated = 0
    total_missing = []
    total_content_sync = []
    total_orphaned = []
    matched_docs = set()
    all_docs_folders = [pair[1] for pair in sync_roots]

    for src_root, docs_root, label, patterns in sync_roots:
        if not src_root.exists():
            print(
                f"Warning: Source directory '{src_root}' not found. Skipping {label}."
            )
            continue

        if not docs_root.exists():
            docs_root.mkdir(parents=True, exist_ok=True)
            print(f"Created documentation directory: {docs_root}")

        src_files = []
        for pattern in patterns:
            src_files.extend(list(src_root.rglob(pattern)))

        total_source_files += len(src_files)
        print(f"\n--- {label} Sync ({len(src_files)} files) ---")

        for src_path in src_files:
            rel_path = src_path.relative_to(src_root)
            if rel_path.stem == "__init__":
                continue

            doc_filename = f"{rel_path.stem}_{rel_path.suffix.lstrip('.')}.md".replace(
                "-", "_"
            )
            doc_path = docs_root / rel_path.parent / doc_filename

            if not doc_path.exists():
                total_missing.append(doc_path)
                continue

            matched_docs.add(doc_path.resolve())

            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            current_hash = get_file_hash(src_path)
            mtime = datetime.fromtimestamp(src_path.stat().st_mtime).strftime(
                "%Y-%m-%d"
            )

            # Update lastmod
            new_content, mod_count = re.subn(
                r"(^lastmod:\s*)\d{4}-\d{2}-\d{2}",
                f"\\g<1>{mtime}",
                content,
                flags=re.MULTILINE,
            )

            # Update or add src_hash
            if "src_hash:" in new_content:
                new_content, hash_count = re.subn(
                    r"(^src_hash:\s*).*$",
                    f"\\g<1>{current_hash}",
                    new_content,
                    flags=re.MULTILINE,
                )
                is_different_hash = hash_count > 0 and current_hash not in content
            else:
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

                status_msg = (
                    f"Updated {doc_path.relative_to(PROJECT_ROOT_DIR)} metadata"
                )
                if is_different_hash:
                    status_msg += " (Hash Change Detected)"
                    total_content_sync.append((src_path, doc_path))

                print(status_msg)
                total_updated += 1

    # Orphan detection across all synced folders
    project_root = Path(PROJECT_ROOT_DIR)
    src_root = project_root / "src"

    for docs_root in all_docs_folders:
        if not docs_root.exists():
            continue
        for doc in docs_root.rglob("*.md"):
            if doc.resolve() in matched_docs or doc.name == "index.md":
                continue

            rel_doc = doc.relative_to(docs_root)
            parts = list(rel_doc.parts)
            filename = rel_doc.name

            name_match = re.match(r"^(.*)_([a-z0-9]+)\.md$", filename)
            if name_match:
                stem, ext = name_match.groups()
                possible_names = [f"{stem}.{ext}", f"{stem.replace('_', '-')}.{ext}"]
            else:
                stem = filename.replace(".md", "")
                possible_names = [stem, stem.lower(), stem.capitalize()]

            found_source = False

            if docs_root.name == "source":
                if parts and parts[0] in ["api", "core", "data", "modelling"]:
                    search_dir = src_root / Path(*parts[1:-1])
                elif parts and parts[0] == "config":
                    search_dir = project_root
                elif parts and parts[0] == "frontend":
                    search_dir = project_root / "static"
                else:
                    search_dir = None
            elif docs_root.name == "scripts":
                search_dir = project_root / "scripts" / Path(*parts[:-1])
            else:
                search_dir = None

            if search_dir:
                for name in possible_names:
                    if (search_dir / name).exists():
                        found_source = True
                        break

            if not found_source:
                total_orphaned.append(doc)

    print("\n--- Overall Sync Summary ---")
    print(f"Total Source Files: {total_source_files}")
    print(f"Updated Metadata: {total_updated}")
    print(f"Missing Docs: {len(total_missing)}")
    print(f"Orphaned Docs: {len(total_orphaned)}")
    print(f"Content Sync Needed: {len(total_content_sync)}")

    if total_content_sync:
        print("\n🚀 ACTION REQUIRED: Sync content for:")
        for py_p, doc_p in total_content_sync:
            print(
                f"- [ ] {doc_p.relative_to(PROJECT_ROOT_DIR)} (source: {py_p.relative_to(PROJECT_ROOT_DIR)})"
            )

    if total_missing:
        print("\n⚠️  MISSING DOCUMENTATION:")
        for doc in total_missing:
            print(f"- {doc.relative_to(PROJECT_ROOT_DIR)}")

    if total_orphaned:
        print("\n🗑️  ORPHANED DOCUMENTATION:")
        for doc in total_orphaned:
            print(f"- {doc.relative_to(PROJECT_ROOT_DIR)}")


if __name__ == "__main__":
    sync_metadata()
