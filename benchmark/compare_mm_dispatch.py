#!/usr/bin/env python3
"""Compare two CSV snapshots produced by snapshot_mm_dispatch.py."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


KEY_FIELDS = ("batch", "m", "n", "k", "layout")
COMPARE_FIELDS = (
    "count",
    "eligible",
    "dispatch_route",
    "triton_kernel",
    "launch_policy",
    "launch_count",
)


def _load(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {tuple(row[field] for field in KEY_FIELDS): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{path} contains duplicate dispatch keys")
    return indexed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--shape-dir",
        type=Path,
        help="also write one Count-aware benchmark YAML per changed layout",
    )
    parser.add_argument(
        "--group-shapes-by-kernel",
        action="store_true",
        help="split the optional benchmark YAMLs by the pre-change kernel",
    )
    parser.add_argument("--fail-if-different", action="store_true")
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    all_keys = sorted(set(before) | set(after))
    differences = []
    transition_counts = Counter()
    weighted_transition_counts = Counter()
    for key in all_keys:
        lhs = before.get(key, {})
        rhs = after.get(key, {})
        changed = [
            field for field in COMPARE_FIELDS if lhs.get(field) != rhs.get(field)
        ]
        if not changed:
            continue
        transition = (
            lhs.get("triton_kernel", "<missing>"),
            rhs.get("triton_kernel", "<missing>"),
        )
        count = int(rhs.get("count") or lhs.get("count") or 0)
        transition_counts[transition] += 1
        weighted_transition_counts[transition] += count
        differences.append(
            {
                **{field: value for field, value in zip(KEY_FIELDS, key)},
                "count": rhs.get("count", lhs.get("count", "")),
                "changed_fields": ",".join(changed),
                **{f"before_{field}": lhs.get(field, "") for field in COMPARE_FIELDS},
                **{f"after_{field}": rhs.get(field, "") for field in COMPARE_FIELDS},
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        *KEY_FIELDS,
        "count",
        "changed_fields",
        *(f"before_{field}" for field in COMPARE_FIELDS),
        *(f"after_{field}" for field in COMPARE_FIELDS),
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(differences)

    print(
        f"compared {len(all_keys)} keys: {len(differences)} changed, "
        f"{len(all_keys) - len(differences)} unchanged"
    )
    for transition, shape_count in transition_counts.most_common():
        print(
            f"  {transition[0]} -> {transition[1]}: "
            f"{shape_count} shapes, Count={weighted_transition_counts[transition]}"
        )
    print(f"wrote {len(differences)} differences to {args.output}")
    if args.shape_dir is not None:
        args.shape_dir.mkdir(parents=True, exist_ok=True)
        for layout in ("NN", "NT"):
            layout_rows = [row for row in differences if row["layout"] == layout]
            groups = {"": layout_rows}
            if args.group_shapes_by_kernel:
                groups.update(
                    {
                        f"_{kernel.removesuffix('_kernel_ppu')}": [
                            row
                            for row in layout_rows
                            if row["before_triton_kernel"] == kernel
                        ]
                        for kernel in sorted(
                            {row["before_triton_kernel"] for row in layout_rows}
                        )
                    }
                )
            for suffix, group_rows in groups.items():
                payload = {
                    "mm": {
                        "shape_desc": "B,M,N,K,Count",
                        "shapes": [
                            [
                                int(row["batch"]),
                                int(row["m"]),
                                int(row["n"]),
                                int(row["k"]),
                                int(row["count"]),
                            ]
                            for row in group_rows
                        ],
                    }
                }
                path = (
                    args.shape_dir
                    / f"dispatch_changed_{layout.lower()}{suffix}.yaml"
                )
                path.write_text(
                    yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
                )
                print(
                    f"wrote {len(group_rows)} {layout} shapes to {path}"
                )
    if differences and args.fail_if_different:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
