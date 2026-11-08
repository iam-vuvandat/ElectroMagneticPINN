import os
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


def run_bundle():
    utils_dir = Path(__file__).resolve().parent
    target_dir = utils_dir.parent
    output_file = utils_dir / "bundle.txt"

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

    module_name = target_dir.name.upper()
    packed_count = 0

    print(f"[INFO] Bắt đầu đóng gói mã nguồn cho module: {module_name}...")

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("=" * 80 + "\n")
        outfile.write(f"          MODULE BUNDLE SOURCE: {module_name}\n")
        outfile.write("=" * 80 + "\n\n")

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

                    outfile.write(f"#" + "=" * 78 + "\n")
                    outfile.write(f"# FILE: {rel_path}\n")
                    outfile.write(f"#" + "=" * 78 + "\n\n")

                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                        outfile.write("\n\n" + "-" * 80 + "\n\n")
                        packed_count += 1
                        print(f"  [+] Đã đóng gói: {rel_path}")
                    except Exception as e:
                        print(f"  [!] Lỗi khi đọc file {rel_path}: {e}")

    print("-" * 60)
    print(
        f"[SUCCESS] Hoàn tất! Đã đóng gói thành công {packed_count} file vào"
        f" {output_file.name}."
    )


if __name__ == "__main__":
    run_bundle()