"""Read-only environment inventory for run manifests and G1 evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PREFIXES = ("rocm", "amd-", "cuda-", "nvidia-")
PROTECTED_NAMES = {"torch", "torchvision", "torchaudio", "triton"}


def _name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def requirement_file() -> str:
    """Select by installed build, even when no GPU is visible."""
    import torch

    return "requirements-rocm.txt" if torch.version.hip else "requirements-cuda.txt"


def _locked_sources(filename: str) -> dict[str, dict[str, str]]:
    result = {}
    current = None
    for line in (ROOT / filename).read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==", 1)
            current = _name(name.split("[", 1)[0])
            result[current] = {"version": version, "index": "not_recorded"}
        elif current and line.strip().startswith("# from "):
            result[current]["index"] = line.strip()[7:]
    return result


def _filesystem(path: Path) -> dict:
    resolved = path.resolve()
    while not resolved.exists():
        resolved = resolved.parent
    proc = subprocess.run(
        ["findmnt", "--json", "--target", str(resolved), "--output", "SOURCE,FSTYPE,TARGET,OPTIONS"],
        text=True, capture_output=True,
    )
    return {
        "requested_path": str(path.resolve()),
        "probe_ancestor": str(resolved),
        "mount": json.loads(proc.stdout)["filesystems"][0] if proc.returncode == 0 else None,
        "reason": None if proc.returncode == 0 else proc.stderr.strip(),
    }


def collect_environment(output_root: str | Path | None = None) -> dict:
    """Collect package, build, device and filesystem metadata without updates.

    Registry source is the source recorded in the selected requirements, accompanied by
    the actual installed version and an explicit match bit. pip metadata does
    not retain registry URLs; install logs and the lock supply that provenance.
    """
    import torch

    filename = requirement_file()
    sources = _locked_sources(filename)
    packages = []
    seen = set()
    for distribution in importlib.metadata.distributions():
        name = _name(distribution.metadata["Name"])
        # Editable setuptools also exposes the source egg-info. Match normal
        # metadata lookup by keeping the first distribution on sys.path.
        if name in seen:
            continue
        seen.add(name)
        locked = sources.get(name)
        direct_url = distribution.read_text("direct_url.json")
        packages.append({
            "name": name,
            "version": distribution.version,
            "install_location": str(distribution.locate_file("")),
            "source": json.loads(direct_url) if direct_url else (locked or {"index": "not_recorded"}),
            "lock_matches": distribution.version == locked["version"] if locked else None,
            "source_evidence": "direct_url.json" if direct_url else (filename + " and installation logs" if locked else "not_recorded"),
        })
    packages.sort(key=lambda item: item["name"])
    protected = {p["name"]: p["version"] for p in packages if p["name"].startswith(PROTECTED_PREFIXES) or p["name"] in PROTECTED_NAMES}
    expected = {name: entry["version"] for name, entry in sources.items()
                if name.startswith(PROTECTED_PREFIXES) or name in PROTECTED_NAMES}
    mismatches = {name: {"expected": version, "actual": protected.get(name)} for name, version in expected.items() if protected.get(name) != version}
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name, "total_memory": props.total_memory, "gcn_arch_name": getattr(props, "gcnArchName", None)})
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT / ".python-version", ROOT / "requirements-rocm.txt", ROOT / "requirements-cuda.txt", ROOT / "pyproject.toml", ROOT / "scripts" / "check_environment.py", Path(__file__)]}
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "wsl": "microsoft" in platform.release().lower(),
        "dependencies": packages,
        "requirements_file": filename,
        "protected_packages": protected,
        "protected_mismatches": mismatches,
        "torch": {"version": str(torch.__version__), "hip": torch.version.hip, "cuda": torch.version.cuda, "cuda_api_available": available, "devices": devices},
        "filesystem": _filesystem(Path(output_root) if output_root else ROOT / "outputs"),
        "git_commit": git.stdout.strip() if git.returncode == 0 else "not_provided",
        "git_dirty": bool(dirty.stdout.strip()),
        "source_file_sha256": hashes,
    }


