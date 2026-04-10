from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
ENTRYPOINT = ROOT / "main.py"
OUTPUT = ROOT / ".kaggle_push" / "script.py"

KAGGLE_PIP_DEPS = ["dm-haiku", "lmdb"]


@dataclass(frozen=True)
class LocalModule:
    module_name: str  # e.g. src.cifar.data_loader
    file_path: Path  # e.g. repo/src/cifar/data_loader.py


def make_install_block() -> str:
    if not KAGGLE_PIP_DEPS:
        return ""

    lines = [
        "# Install extra dependencies for Kaggle",
        "import subprocess, sys",
    ]

    for pkg in KAGGLE_PIP_DEPS:
        lines.append(
            f'subprocess.check_call([sys.executable, "-m", "pip", "install", "{pkg}"])'
        )

    return "\n".join(lines) + "\n\n"


def module_name_to_path(module_name: str) -> Path:
    """
    Convert:
        src.cifar.data_loader -> <repo>/src/cifar/data_loader.py
    """
    if not module_name.startswith("src."):
        raise ValueError(f"Not a local src module: {module_name}")

    rel = module_name.split(".")[1:]  # drop leading 'src'
    return SRC_ROOT.joinpath(*rel).with_suffix(".py")


def strip_future_imports(source: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    to_remove = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for i in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                to_remove.add(i)

    kept = [l for i, l in enumerate(lines, 1) if i not in to_remove]
    return "\n".join(kept).strip() + "\n"


def path_to_module_name(path: Path) -> str:
    """
    Convert:
        <repo>/src/cifar/data_loader.py -> src.cifar.data_loader
    """
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def is_local_src_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.startswith("src.") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module is not None and node.module.startswith("src.")
    return False


def extract_local_src_modules(tree: ast.AST) -> list[str]:
    """
    Return module names referenced via local imports.
    Supports:
      - import src.foo.bar
      - import src.foo.bar as baz
      - from src.foo.bar import thing
    """
    modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("src."):
                modules.append(node.module)

    return modules


def strip_local_src_imports(source: str) -> str:
    """
    Remove all local `src.*` import lines.
    Leaves third-party / stdlib imports untouched.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    to_remove: set[int] = set()

    for node in ast.walk(tree):
        if is_local_src_import(node):
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            for i in range(start, end + 1):
                to_remove.add(i)

    stripped = [line for i, line in enumerate(lines, start=1) if i not in to_remove]
    return "\n".join(stripped).strip() + "\n"


def discover_local_dependencies(module_path: Path) -> list[LocalModule]:
    source = read_text(module_path)
    tree = ast.parse(source, filename=str(module_path))

    deps: list[LocalModule] = []
    for mod_name in extract_local_src_modules(tree):
        dep_path = module_name_to_path(mod_name)
        if not dep_path.exists():
            raise FileNotFoundError(
                f"Could not resolve local import {mod_name!r} -> {dep_path}"
            )
        deps.append(LocalModule(module_name=mod_name, file_path=dep_path))
    return deps


def topo_collect(entry_source: str) -> list[LocalModule]:
    """
    Collect local src modules recursively in dependency order:
    dependencies first, dependents later.
    """
    entry_tree = ast.parse(entry_source, filename=str(ENTRYPOINT))
    initial = extract_local_src_modules(entry_tree)

    ordered: list[LocalModule] = []
    seen: set[str] = set()
    temp: set[str] = set()

    def visit(module_name: str) -> None:
        if module_name in seen:
            return
        if module_name in temp:
            raise RuntimeError(f"Cyclic local import detected at {module_name}")

        temp.add(module_name)
        module_path = module_name_to_path(module_name)
        if not module_path.exists():
            raise FileNotFoundError(
                f"Could not resolve local import {module_name!r} -> {module_path}"
            )

        for dep in discover_local_dependencies(module_path):
            visit(dep.module_name)

        temp.remove(module_name)
        seen.add(module_name)
        ordered.append(LocalModule(module_name=module_name, file_path=module_path))

    for module_name in initial:
        visit(module_name)

    return ordered


def make_module_header(module: LocalModule) -> str:
    border = "#" * 79
    return (
        f"\n{border}\n"
        f"# BEGIN MODULE: {module.module_name}\n"
        f"# SOURCE: {module.file_path.relative_to(ROOT)}\n"
        f"{border}\n\n"
    )


def make_module_footer(module: LocalModule) -> str:
    border = "#" * 79
    return f"\n{border}\n# END MODULE: {module.module_name}\n{border}\n"


def build_bundle() -> str:
    entry_source = read_text(ENTRYPOINT)
    modules = topo_collect(entry_source)

    parts: list[str] = []
    parts.append(
        "# Auto-generated by build_kaggle_script.py\n"
        "# Do not edit this file directly.\n"
    )

    # Install block for Kaggle dependencies
    parts.append("from __future__ import annotations\n\n")
    parts.append(make_install_block())

    # Bundle local modules first
    for module in modules:
        source = read_text(module.file_path)
        source = strip_future_imports(source)
        source = strip_local_src_imports(source)
        parts.append(make_module_header(module))
        parts.append(source.rstrip() + "\n")
        parts.append(make_module_footer(module))

    # Then entrypoint without local src imports
    entry_source = strip_local_src_imports(entry_source)
    parts.append("\n" + "#" * 79 + "\n# ENTRYPOINT\n" + "#" * 79 + "\n\n")
    parts.append(entry_source.rstrip() + "\n")

    return "".join(parts)


def ensure_output_dir() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    bundle = build_bundle()
    ensure_output_dir()
    OUTPUT.write_text(bundle, encoding="utf-8")
    print(f"Wrote bundled script to: {OUTPUT}")


if __name__ == "__main__":
    main()
