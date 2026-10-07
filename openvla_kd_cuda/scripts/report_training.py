"""Render saved training/held-out validation logs; loss is not task success."""
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    # A running writer may be midway through the last JSONL record.
    if text and not text.endswith("\n"):
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines if line.strip()]
    if any(not math.isfinite(v) for row in rows for v in row.values() if isinstance(v, (int, float))):
        raise ValueError("Non-finite metric found")
    return rows


def render(folder):
    cache = Path(__file__).resolve().parents[1] / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    env_path = folder / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    rows = read_rows(folder / "metrics.jsonl")
    val = read_rows(folder / "validation.jsonl")
    if not rows:
        raise ValueError("No complete training updates to report")
    steps = [r["step"] for r in rows]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Duplicate or out-of-order training steps")
    keys = ["step", "demo", "action", "feature", "total", "grad_norm", "seconds"]
    with (folder / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in keys} for r in rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    if env.get("note"):
        fig.suptitle(env["note"], fontsize=10)
    for key in ("total", "demo", "action", "feature"):
        if key in rows[0]: axes[0, 0].plot(steps, [r[key] for r in rows], label=key, linewidth=0.8, alpha=0.8)
    axes[0, 0].set_title("Training loss; components shown unweighted")
    axes[0, 0].legend()
    if val:
        axes[0, 1].plot([r["step"] for r in val], [r["validation_l1"] for r in val], marker="o")
    axes[0, 1].set_title("Held-out action L1 (lower is better)")
    axes[1, 0].plot(steps, [r["grad_norm"] for r in rows])
    axes[1, 0].set_title("Gradient norm before clipping")
    axes[1, 1].plot(steps, [r["seconds"] for r in rows])
    axes[1, 1].set_title("Seconds per update (excludes validation / saving)")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.grid(alpha=0.2)
    fig.savefig(folder / "curves.png", dpi=140)
    plt.close(fig)
    metadata = {"completed_updates_in_this_directory": len(rows), "last_step": steps[-1],
                "latest_validation_l1": val[-1]["validation_l1"] if val else None,
                "update_seconds_total": sum(r["seconds"] for r in rows)}
    for name in ("status", "summary", "environment"):
        p = folder / (name + ".json")
        if p.exists(): metadata[name] = json.loads(p.read_text())
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>2B 训练日志</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}img{width:100%}'
            'pre{white-space:pre-wrap;background:#f2f4f7;padding:18px}</style><h1>训练与离线验证</h1>'
            '<p>'+html.escape(env.get("note", ""))+'</p>'
            '<p>损失曲线用于检查学习过程。机器人任务成功率需要另做闭环仿真，不能从本图推断。</p>'
            '<p><a href="metrics.csv">训练 CSV</a> · <a href="metrics.jsonl">原始训练日志</a> · '
            '<a href="validation.jsonl">验证日志</a></p><img src="curves.png" alt="训练、验证、梯度和耗时曲线">'
            '<pre>' + html.escape(json.dumps(metadata, ensure_ascii=False, indent=2)) + '</pre></html>')
    (folder / "report.html").write_text(page)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder")
    print(json.dumps(render(p.parse_args().folder), ensure_ascii=False, indent=2))
