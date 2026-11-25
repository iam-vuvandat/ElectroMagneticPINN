import os
import sys
import subprocess
from pathlib import Path


def load_gitignore_patterns(target_dir):
    gitignore_file = target_dir / ".gitignore"
    patterns = set()
    if gitignore_file.exists():
        with open(gitignore_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.add(line.replace("\\", "/").strip("/"))
    return patterns


def normalize_excludes(items):
    if not items:
        return set()
    return {str(item).replace("\\", "/").strip("/") for item in items if item}


def is_excluded(rel_path_str, name, exclude_set):
    return name in exclude_set or rel_path_str in exclude_set


def generate_tree(
    target_dir,
    ignore_dirs,
    ignore_files,
    gitignore_patterns,
):
    tree_lines = [target_dir.name + "/"]
    valid_extensions = {
        ".py",
        ".md",
        ".txt",
        ".tex",
        ".yml",
        ".yaml",
        ".json",
        ".sh",
        ".ipynb",
        ".c",
        ".cpp",
        ".h",
    }

    def walk(current_path, prefix=""):
        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            return

        filtered = []
        for entry in entries:
            full_path = current_path / entry
            rel_path_str = full_path.relative_to(target_dir).as_posix()
            if full_path.is_dir():
                if (
                    entry.startswith(".")
                    or is_excluded(rel_path_str, entry, ignore_dirs)
                ):
                    continue
            else:
                if (
                    is_excluded(rel_path_str, entry, ignore_files)
                    or is_excluded(rel_path_str, entry, gitignore_patterns)
                    or full_path.suffix not in valid_extensions
                ):
                    continue
            filtered.append(entry)

        for i, entry in enumerate(filtered):
            full_path = current_path / entry
            is_last = i == len(filtered) - 1
            tree_lines.append(prefix + ("└── " if is_last else "├── ") + entry)
            if full_path.is_dir():
                walk(full_path, prefix + ("    " if is_last else "│   "))

    walk(target_dir)
    return "\n".join(tree_lines)


def copy_file_to_clipboard(file_path):
    if sys.platform == "win32":
        abs_path = str(Path(file_path).resolve())
        ps_script = f"Set-Clipboard -Path '{abs_path}'"
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                check=True
            )
            print(f"[*] Successfully copied file into Windows Clipboard: {abs_path}")
        except Exception as e:
            print(f"[!] Failed to copy file to clipboard: {e}")


def run_bundle(exclude_files=None, exclude_dirs=None):
    utils_dir = Path(__file__).resolve().parent
    target_dir = utils_dir.parent
    output_file = utils_dir / "bundle.txt"

    valid_extensions = {
        ".py",
        ".md",
        ".txt",
        ".tex",
        ".yml",
        ".yaml",
        ".json",
        ".sh",
        ".ipynb",
        ".c",
        ".cpp",
        ".h",
    }
    default_ignore_dirs = {
        "__pycache__",
        ".git",
        ".idea",
        ".vscode",
        ".pytest_cache",
        "venv",
        "env",
        "build",
        "dist",
        ".ipynb_checkpoints",
        "node_modules",
        "utils",
    }
    default_ignore_files = {output_file.name, Path(__file__).name, "batch.log"}

    gitignore_patterns = load_gitignore_patterns(target_dir)
    ignore_dirs = (
        default_ignore_dirs
        | gitignore_patterns
        | normalize_excludes(exclude_dirs)
    )
    ignore_files = (
        default_ignore_files
        | gitignore_patterns
        | normalize_excludes(exclude_files)
    )

    print(f"Packing module: {target_dir.name.upper()}...")

    tree_str = generate_tree(
        target_dir,
        ignore_dirs,
        ignore_files,
        gitignore_patterns,
    )

    bundle_content = []
    bundle_content.append(f"MODULE BUNDLE SOURCE: {target_dir.name.upper()}\n\n")
    bundle_content.append("DIRECTORY STRUCTURE:\n")
    bundle_content.append(tree_str + "\n\n")

    for root, dirs, files in os.walk(target_dir):
        root_path = Path(root)
        kept_dirs = []
        for d in sorted(dirs):
            dir_rel_str = (root_path / d).relative_to(target_dir).as_posix()
            if not d.startswith(".") and not is_excluded(dir_rel_str, d, ignore_dirs):
                kept_dirs.append(d)
        dirs[:] = kept_dirs

        for file_name in sorted(files):
            file_path = root_path / file_name
            rel_path = file_path.relative_to(target_dir)
            rel_path_str = rel_path.as_posix()
            if is_excluded(rel_path_str, file_path.name, ignore_files):
                continue
            if file_path.suffix in valid_extensions:
                bundle_content.append(f"FILE: {rel_path}\n\n")
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        bundle_content.append(infile.read())
                    bundle_content.append("\n\n")
                except Exception as e:
                    print(f"Error reading {rel_path}: {e}")

    final_text = "".join(bundle_content)

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(final_text)

    copy_file_to_clipboard(output_file)


if __name__ == "__main__":
    exclude_files = ["bundle.txt","requirements.txt","update.py","README.md"]
    exclude_dirs = ["utils","ansys_aedt","animation_frames","checkpoints"]
    run_bundle(exclude_files=exclude_files, exclude_dirs=exclude_dirs)