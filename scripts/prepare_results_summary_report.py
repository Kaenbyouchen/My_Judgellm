#!/usr/bin/env python3
"""
Normalize outputs/results_summary.csv for reporting:
- fill missing acc using dataset+judge group value
- sort rows by dataset -> bias -> model
- generate report charts (PNG only, no extra CSV artifacts)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


CSV_FIELDS = [
    "data type (category)",
    "dataset",
    "bias type",
    "bias injection mode",
    "data type (pairwise/scalar)",
    "judge llm",
    "acc",
    "acc_biased",
    "cr",
    "rr",
]


def _to_float_percent(text: str):
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Keep only known fields and normalize missing keys.
    normalized = []
    for row in rows:
        item = {k: str(row.get(k, "")).strip() for k in CSV_FIELDS}
        normalized.append(item)

    # Fill missing acc by dataset+judge group.
    acc_ref = {}
    for row in normalized:
        key = (row["dataset"], row["judge llm"])
        if row["acc"] and key not in acc_ref:
            acc_ref[key] = row["acc"]
    for row in normalized:
        if not row["acc"]:
            row["acc"] = acc_ref.get((row["dataset"], row["judge llm"]), "")

    # De-duplicate same evaluation key, keep the latest row in file order.
    dedup = {}
    for row in normalized:
        key = (
            row["dataset"],
            row["bias type"],
            row["bias injection mode"],
            row["data type (pairwise/scalar)"],
            row["judge llm"],
        )
        dedup[key] = row
    normalized = list(dedup.values())

    normalized.sort(
        key=lambda r: (
            r["dataset"],
            r["bias type"],
            r["judge llm"],
            r["bias injection mode"],
            r["data type (category)"],
            r["data type (pairwise/scalar)"],
        )
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)

    return normalized


def aggregate_bias_metrics(rows: list[dict]) -> list[tuple[str, float, float, float, float]]:
    by_bias = defaultdict(lambda: {"acc": [], "acc_biased": [], "cr": [], "rr": []})
    for row in rows:
        bias = row["bias type"] or "unknown"
        for k in ("acc", "acc_biased", "cr", "rr"):
            v = _to_float_percent(row.get(k, ""))
            if v is not None:
                by_bias[bias][k].append(v)

    data = []
    for bias in sorted(by_bias.keys()):
        m = by_bias[bias]
        data.append(
            (
                bias,
                mean(m["acc"]) if m["acc"] else 0.0,
                mean(m["acc_biased"]) if m["acc_biased"] else 0.0,
                mean(m["cr"]) if m["cr"] else 0.0,
                mean(m["rr"]) if m["rr"] else 0.0,
            )
        )

    return data


def aggregate_metric(rows: list[dict], group_key: str, metric: str) -> list[tuple[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get(group_key) or "").strip() or "unknown"
        v = _to_float_percent(row.get(metric, ""))
        if v is not None:
            grouped[key].append(v)
    data = []
    for key in sorted(grouped.keys()):
        vals = grouped[key]
        data.append((key, mean(vals) if vals else 0.0))
    return data


def try_plot_metric_bar(data: list[tuple[str, float]], out_png: Path, title: str, y_label: str) -> bool:
    try:
        import matplotlib.pyplot as plt

        if not data:
            return False
        # Sort descending so cross-bar difference is visually clearer.
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        x = [k for k, _ in data_sorted]
        y = [v for _, v in data_sorted]

        ymin = max(0.0, min(y) - 5.0)
        ymax = min(100.0, max(y) + 5.0)
        if ymax - ymin < 10.0:
            mid = (ymax + ymin) / 2.0
            ymin = max(0.0, mid - 5.0)
            ymax = min(100.0, mid + 5.0)

        plt.figure(figsize=(13, 6))
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(x))]
        bars = plt.bar(x, y, color=colors, edgecolor="black", linewidth=0.8)
        plt.xticks(rotation=28, ha="right")
        plt.ylabel(y_label)
        plt.title(title)
        plt.ylim(ymin, ymax)
        plt.grid(axis="y", linestyle="--", alpha=0.35)

        # Label exact value on each bar.
        for b, v in zip(bars, y):
            plt.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + (ymax - ymin) * 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        return True
    except Exception:
        return False


def try_plot(data: list[tuple[str, float, float, float, float]], out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        biases = [x[0] for x in data]
        acc = [x[1] for x in data]
        acc_b = [x[2] for x in data]
        cr = [x[3] for x in data]
        rr = [x[4] for x in data]

        plt.figure(figsize=(12, 6))
        plt.plot(biases, acc, marker="o", label="acc(mean)")
        plt.plot(biases, acc_b, marker="o", label="acc_biased(mean)")
        plt.plot(biases, cr, marker="o", label="cr(mean)")
        plt.plot(biases, rr, marker="o", label="rr(mean)")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Percent")
        plt.title("Metrics by Bias (Mean over rows in results_summary.csv)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        return True
    except Exception:
        # Fall back to Pillow bar chart if matplotlib is unavailable/unstable.
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return False

        width, height = 1400, 800
        margin_l, margin_r, margin_t, margin_b = 100, 40, 70, 180
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b
        bg = (255, 255, 255)
        axis = (30, 30, 30)
        colors = {
            "acc": (52, 152, 219),
            "acc_biased": (46, 204, 113),
            "cr": (241, 196, 15),
            "rr": (231, 76, 60),
        }

        img = Image.new("RGB", (width, height), bg)
        d = ImageDraw.Draw(img)

        # Axis
        x0, y0 = margin_l, height - margin_b
        x1, y1 = width - margin_r, margin_t
        d.line([(x0, y0), (x1, y0)], fill=axis, width=2)
        d.line([(x0, y0), (x0, y1)], fill=axis, width=2)
        d.text((margin_l, 20), "Metrics by Bias (mean, %)", fill=axis)

        # Y ticks (0~100)
        for p in range(0, 101, 20):
            y = y0 - (p / 100.0) * plot_h
            d.line([(x0 - 5, y), (x0, y)], fill=axis, width=1)
            d.text((x0 - 35, y - 7), f"{p}", fill=axis)

        n = len(data)
        if n == 0:
            img.save(out_png)
            return True

        group_w = plot_w / n
        bar_w = max(6, int(group_w * 0.16))
        offsets = [-1.5, -0.5, 0.5, 1.5]
        labels = ["acc", "acc_biased", "cr", "rr"]

        for i, (bias, acc, acc_b, cr, rr) in enumerate(data):
            cx = x0 + group_w * (i + 0.5)
            values = [acc, acc_b, cr, rr]
            for off, lab, v in zip(offsets, labels, values):
                bx = int(cx + off * bar_w)
                by = int(y0 - (max(0.0, min(100.0, v)) / 100.0) * plot_h)
                d.rectangle([(bx, by), (bx + bar_w - 1, y0)], fill=colors[lab], outline=colors[lab])
            d.text((int(cx - group_w * 0.35), y0 + 10), bias, fill=axis)

        # Legend
        lx, ly = margin_l, height - 120
        for lab in labels:
            d.rectangle([(lx, ly), (lx + 18, ly + 12)], fill=colors[lab], outline=colors[lab])
            d.text((lx + 24, ly - 2), f"{lab}(mean)", fill=axis)
            lx += 200

        img.save(out_png)
        return True


def main():
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "outputs" / "results_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    rows = normalize_csv(csv_path)
    data = aggregate_bias_metrics(rows)

    out_png = root / "outputs" / "results_summary_by_bias.png"
    plotted = try_plot(data, out_png)

    print(f"Normalized CSV: {csv_path}")
    if plotted:
        print(f"Bias chart PNG: {out_png}")
    else:
        print("Bias chart PNG skipped: matplotlib unavailable")

    # Additional report charts (acc/rr only):
    # - compare models
    # - compare biases
    # - compare datasets
    jobs = [
        ("judge llm", "model", "acc"),
        ("judge llm", "model", "rr"),
        ("bias type", "bias", "acc"),
        ("bias type", "bias", "rr"),
        ("dataset", "dataset", "acc"),
        ("dataset", "dataset", "rr"),
    ]
    for group_key, short_name, metric in jobs:
        d = aggregate_metric(rows, group_key=group_key, metric=metric)
        out_png = root / "outputs" / f"results_summary_{metric}_by_{short_name}.png"
        ok = try_plot_metric_bar(
            d,
            out_png=out_png,
            title=f"{metric.upper()} mean by {short_name}",
            y_label="Percent",
        )
        if ok:
            print(f"{metric.upper()} by {short_name} PNG: {out_png}")
        else:
            print(f"{metric.upper()} by {short_name} PNG skipped: matplotlib unavailable")


if __name__ == "__main__":
    main()

