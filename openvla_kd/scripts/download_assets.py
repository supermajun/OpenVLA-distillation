"""Download only public, pinned research inputs; no authentication or upload."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

STUDENT_REV = "a7da5b986cb59b408707209984f360a5f4ad7e47"
TEACHER_REV = "6d0231af0e48c5985f1ff86908f4674b84bc049b"
DATA_REV = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    partial = dest.with_name(dest.name + ".part")
    print(f"Downloading {dest}", flush=True)
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "4", "--continue-at", "-", "--output", str(partial), url], check=True)
    partial.replace(dest)
    print(f"Ready {dest} ({dest.stat().st_size:,} bytes)", flush=True)


def model_jobs(repo, rev, dest):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}") as f:
        info = json.load(f)
    jobs = []
    for item in info["siblings"]:
        name = item["rfilename"]
        if "/" not in name and name.endswith((".json", ".safetensors", ".model", ".txt", ".pt", ".py")):
            jobs.append((f"https://huggingface.co/{repo}/resolve/{rev}/{name}", Path(dest) / name))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("asset", choices=["student", "teacher", "data"])
    args = p.parse_args()
    if args.asset == "student":
        jobs = model_jobs("HuggingFaceTB/SmolVLM-500M-Instruct", STUDENT_REV, "models/student")
    elif args.asset == "teacher":
        jobs = model_jobs("moojink/openvla-7b-oft-finetuned-libero-spatial", TEACHER_REV, "models/teacher")
    else:
        repo = "yifengzhu-hf/LIBERO-datasets"
        rev = DATA_REV
        name = "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
        dest = Path("data/raw")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "source.json").write_text(json.dumps({"repo": repo, "revision": rev, "file": name}, indent=2))
        jobs = [(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}", dest / name)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))


if __name__ == "__main__":
    main()
