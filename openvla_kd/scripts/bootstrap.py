"""Recreate a project-local environment and pinned upstream code; no global installs."""
import os
from pathlib import Path
import subprocess

REPOS = {
    "openvla-oft": ("https://github.com/moojink/openvla-oft.git", "e4287e94541f459edc4feabc4e181f537cd569a8"),
    "LIBERO": ("https://github.com/Lifelong-Robot-Learning/LIBERO.git", "8f1084e3132a39270c3a13ebe37270a43ece2a01"),
}


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    env = {**os.environ, "UV_CACHE_DIR": str(root / ".cache/uv"),
           "UV_PYTHON_INSTALL_DIR": str(root / ".cache/python")}
    if not (root / ".venv/bin/python").exists():
        subprocess.run(["uv", "venv", "--python", "3.11", ".venv"], check=True, env=env)
    subprocess.run(["uv", "pip", "sync", "--python", ".venv/bin/python", "requirements.lock.txt"], check=True, env=env)
    (root / "vendor").mkdir(exist_ok=True)
    for name, (url, revision) in REPOS.items():
        dest = root / "vendor" / name
        if dest.exists():
            actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
            if actual != revision:
                raise RuntimeError(f"Existing {name} is at {actual}, expected {revision}; not overwriting it")
        else:
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", url, str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", revision], check=True)


if __name__ == "__main__":
    main()
