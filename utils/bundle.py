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
                    patterns.add(line.rstrip("/"))
    return patterns


def generate_tree(
    target_dir,
    ignore_dirs,
    gitignore_patterns,
    output_file_name,
    current_script_name,
):
    tree_lines = [target_dir.name + "/"]
    valid_extensions = {
        ".py",
        ".md",
        ".txt",
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
            if full_path.is_dir():
                if entry in ignore_dirs or entry.startswith("."):
                    continue
            else:
                if (
                    entry in {output_file_name, current_script_name, "batch.log"}
                    or entry in gitignore_patterns
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


def run_bundle():
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
    ignore_dirs = {
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

    gitignore_patterns = load_gitignore_patterns(target_dir)
    ignore_dirs.update(gitignore_patterns)
    ignore_files = {output_file.name, Path(__file__).name, "batch.log"}

    print(f"Packing module: {target_dir.name.upper()}...")

    tree_str = generate_tree(
        target_dir,
        ignore_dirs,
        gitignore_patterns,
        output_file.name,
        Path(__file__).name,
    )

    bundle_content = []
    bundle_content.append(f"MODULE BUNDLE SOURCE: {target_dir.name.upper()}\n\n")
    bundle_content.append("DIRECTORY STRUCTURE:\n")
    bundle_content.append(tree_str + "\n\n")

    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [
            d
            for d in dirs
            if d not in ignore_dirs and not d.startswith(".")
        ]
        for file_name in sorted(files):
            file_path = Path(root) / file_name
            if (
                file_path.name in ignore_files
                or file_path.name in gitignore_patterns
            ):
                continue
            if file_path.suffix in valid_extensions:
                rel_path = file_path.relative_to(target_dir)
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
    run_bundle()