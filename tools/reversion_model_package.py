#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assign a release version to a trained FlagTune package, preserving its trees.

Run in an environment with FlagTree and PyYAML installed. The output directory
must be new. Both the platform package and its children receive the requested
version, including the config digest bound into each XGBoost model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def reversion(source, source_version, version, output, platform_key):
    import yaml
    from triton.flagtune.contract.archive import (
        platform_package_name,
        read_model_archive_bytes,
        read_platform_package,
        validate_model_version,
        write_model_archive,
        write_platform_package,
    )
    from triton.flagtune.contract.operator_schema import (
        model_config_sha256,
        model_identity_from_config,
    )

    version = validate_model_version(version)
    if output.exists():
        raise FileExistsError(output)
    package = read_platform_package(
        source, expected_platform_key=platform_key, expected_version=source_version
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reversion-", dir=output.parent) as tmp:
        stage = Path(tmp) / "release"
        children, records = {}, []
        for key, payload in sorted(package.archives.items()):
            members = read_model_archive_bytes(payload)
            config = yaml.safe_load(members["flagtune_config.yaml"])
            summary = json.loads(members["training_summary.json"])
            model = json.loads(members["xgboost_ranker.json"])
            original = copy.deepcopy(model)
            old_digest = model_config_sha256(config)
            assert (
                model["learner"]["attributes"]["flagtune_config_sha256"] == old_digest
            )
            assert summary["model_config_sha256"] == old_digest
            assert summary["model_version"] == config["model_version"]

            config["model_version"] = version
            new_digest = model_config_sha256(config)
            summary.update(model_version=version, model_config_sha256=new_digest)
            model["learner"]["attributes"]["flagtune_config_sha256"] = new_digest
            restored = copy.deepcopy(model)
            restored["learner"]["attributes"]["flagtune_config_sha256"] = old_digest
            assert restored == original
            members.update(
                {
                    "flagtune_config.yaml": yaml.safe_dump(
                        config, sort_keys=False, allow_unicode=True
                    ).encode(),
                    "training_summary.json": json.dumps(summary, indent=2).encode(),
                    "xgboost_ranker.json": json.dumps(
                        model, separators=(",", ":"), allow_nan=False
                    ).encode(),
                }
            )
            identity = model_identity_from_config(config)
            path = stage / "children" / key / "model.tar.gz"
            write_model_archive(path, members)
            saved = json.loads(
                read_model_archive_bytes(path.read_bytes())["xgboost_ranker.json"]
            )
            saved["learner"]["attributes"]["flagtune_config_sha256"] = old_digest
            assert saved == original
            children[identity] = path
            records.append({"identity": key, "sha256": digest(path.read_bytes())})

        archive = stage / "package" / platform_package_name(platform_key, version)
        write_platform_package(
            archive,
            platform_key=platform_key,
            package_version=version,
            model_archives=children,
            required_identities=children,
        )
        report = {
            "version": version,
            "platform_key": platform_key,
            "source_version": source_version,
            "source_package_sha256": digest(source.read_bytes()),
            "package_filename": archive.name,
            "sha256": digest(archive.read_bytes()),
            "models": len(children),
            "all_model_data_unchanged_except_config_binding": True,
            "model_records": records,
        }
        (stage / "package-info.json").write_text(json.dumps(report, indent=2) + "\n")
        os.rename(stage, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", default="thead-zw810e")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reversion(
        args.source, args.source_version, args.version, args.output, args.platform
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "model_records"}, indent=2)
    )


if __name__ == "__main__":
    main()
