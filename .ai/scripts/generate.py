"""Generate Codex instructions and skills from .ai; never adopt unknown files."""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import tempfile

MANIFEST = ".ai/generated-manifest.json"
LIMIT = 32 * 1024
SKIP = {".git", ".venv", "__pycache__", "node_modules", ".claude", ".codex"}


class Conflict(ValueError):
    """A source or destination cannot safely be managed."""


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or str(path) != value
            or ".." in path.parts or "\\" in value or "\0" in value):
        raise Conflict(f"Unsafe path: {value!r}")
    return path


def boundaries(root: Path) -> set[str]:
    config = configparser.ConfigParser(interpolation=None)
    if (root / ".gitmodules").exists():
        config.read(root / ".gitmodules")
    return {str(relative_path(config[s]["path"])) for s in config.sections()
            if "path" in config[s]}


def safe_path(root: Path, name: str) -> Path:
    parts = relative_path(name).parts
    external = boundaries(root)
    current = root
    for i, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            raise Conflict(f"Symlink is outside generator ownership: {name}")
        if "/".join(parts[:i + 1]) in external or (current / ".git").exists():
            raise Conflict(f"Nested repository is outside generator ownership: {name}")
    return current


def source_files(root: Path, base: str) -> list[Path]:
    start = safe_path(root, base)
    result = []
    for directory, dirs, files in os.walk(start, followlinks=False):
        for name in dirs[:]:
            if name in SKIP:
                dirs.remove(name)
            else:
                safe_path(root, (Path(directory) / name).relative_to(root).as_posix())
        for name in files:
            if name == ".DS_Store" or name.startswith(".env") or name.endswith(".pyc"):
                continue
            path = safe_path(root, (Path(directory) / name).relative_to(root).as_posix())
            if not path.is_file():
                raise Conflict(f"Not a regular source file: {path}")
            result.append(path)
    return sorted(result)


def text(data: bytes) -> str:
    return data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


def split_header(value: str) -> tuple[str, str]:
    if not value.startswith("---\n"):
        return "", value
    match = re.search(r"^---[ \t]*$", value[4:], re.M)
    if match is None:
        raise Conflict("Unclosed frontmatter")
    end = 4 + match.end()
    return value[:end], value[end:].lstrip("\n")


def rule(value: str) -> tuple[list[str], str]:
    header, body = split_header(value)
    lines = header.splitlines()[1:-1]
    if not lines or lines[0] != "paths:":
        raise Conflict("Rules require paths frontmatter with a quoted list")
    patterns = []
    for line in lines[1:]:
        if not line.strip():
            continue
        match = re.fullmatch(r'  - (".*")', line)
        if not match:
            raise Conflict(f"Unsupported paths syntax: {line}")
        pattern = json.loads(match[1])
        if not isinstance(pattern, str):
            raise Conflict("Rule paths must be strings")
        relative_path(pattern)
        patterns.append(pattern)
    if not patterns or not body.strip():
        raise Conflict("Rules need paths and a nonempty body")
    return list(dict.fromkeys(patterns)), body.strip()


def directory_for(root: Path, pattern: str) -> str:
    directories = []
    for part in relative_path(pattern).parts[:-1]:
        if any(c in part for c in "*?[{"):
            break
        directories.append(part)
    if not directories or directories[0] in {".ai", ".agents", "references"} | SKIP:
        return "."
    directory = "/".join(directories)
    # A rule may cover a repository boundary, but must never write inside it.
    try:
        safe_path(root, directory + "/AGENTS.md")
    except Conflict:
        return "."
    return directory


def generated(data: bytes, source: str) -> bytes:
    header, body = split_header(text(data))
    notice = f"<!-- GENERATED from {source}; edit .ai/ sources and run uv run python .ai/scripts/generate.py. -->"
    return ((header + "\n" if header else "") + notice + "\n\n"
            + body.rstrip() + "\n").encode()


def relocate_links(body: str, source: str, target: str) -> str:
    def replace(match: re.Match) -> str:
        url = match[1]
        if url.startswith(("#", "/")) or re.match(r"[\w+.-]+:", url):
            return match[0]
        path, separator, anchor = url.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        relative_path(resolved)
        link = posixpath.relpath(resolved, posixpath.dirname(target) or ".")
        return "](" + link + (separator + anchor if separator else "") + ")"
    return re.sub(r"\]\(([^\s()]+)\)", replace, body)


def desired(root: Path) -> dict[str, tuple[bytes, int]]:
    common = text(safe_path(root, ".ai/INSTRUCTIONS.md").read_bytes()).strip()
    if not common:
        raise Conflict("Empty .ai/INSTRUCTIONS.md")
    groups: dict[str, list[str]] = {".": []}
    outputs = {}
    for path in source_files(root, ".ai/rules"):
        if path.suffix != ".md":
            raise Conflict(f"Rules must be Markdown: {path}")
        source = path.relative_to(root).as_posix()
        patterns, body = rule(text(path.read_bytes()))
        local: dict[str, list[str]] = {}
        for pattern in patterns:
            local.setdefault(directory_for(root, pattern), []).append(pattern)
        for directory, conditions in local.items():
            target = "AGENTS.md" if directory == "." else directory + "/AGENTS.md"
            adapted = relocate_links(body, source, target)
            adapted = re.sub(r"^(#{1,5}) ", r"#\1 ", adapted, flags=re.M)
            groups.setdefault(directory, []).append(
                "适用条件（相对于仓库根）：" + "、".join(f"`{p}`" for p in conditions)
                + "。\n\n" + adapted)
    for directory, bodies in groups.items():
        target = "AGENTS.md" if directory == "." else directory + "/AGENTS.md"
        content = common if directory == "." else (
            f"# {directory} 工作要求\n\n叠加仓库根到本目录的全部祖先 AGENTS.md；"
            "下列要求仅在所列路径条件命中时执行。")
        if bodies:
            content += "\n\n" + "\n\n".join(bodies)
        outputs[target] = (generated(content.encode(), ".ai/INSTRUCTIONS.md + .ai/rules/*"), 0o644)
    for path in source_files(root, ".ai/skills"):
        relative = path.relative_to(root / ".ai/skills")
        if len(relative.parts) < 2:
            raise Conflict("Skill files must live in a named skill directory")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", relative.parts[0]):
            raise Conflict(f"Invalid skill name: {relative.parts[0]}")
        if not safe_path(root, f".ai/skills/{relative.parts[0]}/SKILL.md").is_file():
            raise Conflict(f"Skill lacks SKILL.md: {relative.parts[0]}")
        data = path.read_bytes()
        if path.suffix == ".md":
            data = generated(data, path.relative_to(root).as_posix())
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        outputs[".agents/skills/" + relative.as_posix()] = data, mode
    for name in outputs:
        if name.endswith("AGENTS.md") and not name.startswith(".agents/"):
            directory = PurePosixPath(name).parent
            ancestors = {directory, *directory.parents}
            size = sum(len(data) for p, (data, _) in outputs.items()
                       if PurePosixPath(p).name == "AGENTS.md"
                       and not p.startswith(".agents/")
                       and PurePosixPath(p).parent in ancestors)
            if size > LIMIT:
                raise Conflict(f"AGENTS chain exceeds {LIMIT} bytes: {name} ({size})")
    return outputs


def managed(name: str) -> bool:
    path = relative_path(name)
    return ((path.name == "AGENTS.md" and path.parts[0] not in {".ai", ".agents", "references"} | SKIP)
            or (name.startswith(".agents/skills/") and len(path.parts) >= 4))


def record(data: bytes, mode: int) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "mode": mode}


def snapshot(path: Path) -> tuple[bytes, int] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise Conflict(f"Expected regular file: {path}")
    return path.read_bytes(), stat.S_IMODE(path.stat().st_mode)


def reject_unknown_entries(root: Path, owned: set[str]) -> None:
    external = boundaries(root)
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        dirs[:] = [d for d in dirs if d not in SKIP | {".ai", "references"}
                   and not (base / d).is_symlink() and not (base / d / ".git").exists()
                   and (base / d).relative_to(root).as_posix() not in external]
        for name in ("AGENTS.md", "AGENTS.override.md"):
            path = base / name
            if name in files or path.is_symlink() or name in dirs:
                relative = path.relative_to(root).as_posix()
                if relative not in owned:
                    raise Conflict(f"Unmanaged native instruction: {relative}")


def atomic(path: Path, value: tuple[bytes, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".ai-gen-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value[0])
            os.fchmod(stream.fileno(), value[1])
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def generate(root: Path, *, check: bool = False) -> bool:
    root = root.resolve()
    manifest_path = safe_path(root, MANIFEST)
    old_manifest = snapshot(manifest_path)
    previous = {}
    if old_manifest:
        document = json.loads(old_manifest[0])
        if document.get("version") != 1 or not isinstance(document.get("files"), dict):
            raise Conflict("Invalid generated manifest")
        previous = document["files"]
    for name, metadata in previous.items():
        if (not managed(name) or not isinstance(metadata, dict)
                or set(metadata) != {"sha256", "mode"}
                or metadata["mode"] not in (0o644, 0o755)
                or not isinstance(metadata["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])):
            raise Conflict(f"Invalid manifest entry: {name}")
    outputs = desired(root)
    reject_unknown_entries(root, set(previous))
    before = {}
    for name in sorted(set(previous) | set(outputs)):
        path = safe_path(root, name)
        current = snapshot(path)
        if current is not None:
            if name not in previous:
                raise Conflict(f"Unmanaged output: {name}")
            if record(*current) != previous[name]:
                raise Conflict(f"Generated file was edited: {name}; preserve it and reconcile the source")
        before[name] = current
    expected = {name: record(*value) for name, value in sorted(outputs.items())}
    manifest = ((json.dumps({"version": 1, "files": expected}, ensure_ascii=False,
                           indent=2, sort_keys=True) + "\n").encode(), 0o644)
    changes = [name for name in before if before[name] != outputs.get(name)]
    if check:
        for name in changes:
            print(("UPDATE " if name in outputs else "DELETE ") + name)
        if old_manifest != manifest:
            print("UPDATE " + MANIFEST)
        print(f"{len(changes)} output changes; manifest change: {old_manifest != manifest}")
        return not changes and old_manifest == manifest
    applied = []
    try:
        for name in changes:
            path = safe_path(root, name)
            if snapshot(path) != before[name]:
                raise Conflict(f"Output changed during generation: {name}")
            applied.append(name)
            if name in outputs:
                atomic(path, outputs[name])
            else:
                path.unlink()
        if snapshot(manifest_path) != old_manifest:
            raise Conflict("Manifest changed during generation")
        if old_manifest != manifest:
            atomic(manifest_path, manifest)
    except BaseException:
        for name in reversed(applied):
            path = safe_path(root, name)
            if before[name] is None:
                path.unlink(missing_ok=True)
            else:
                atomic(path, before[name])
        raise
    print(f"Generated {len(outputs)} Codex files; changed {len(changes)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        return 0 if generate(Path(__file__).resolve().parents[2], check=args.check) else 1
    except (Conflict, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
