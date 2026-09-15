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

from pathlib import Path

import pytest

from flag_gems.runtime.configs_loader import TunedConfigLoader


YAML_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flag_gems"
    / "runtime"
    / "backend"
    / "_thead"
    / "mm_ppu_expand.yaml"
)


@pytest.mark.parametrize(
    ("op_name", "strategy_size", "required_meta"),
    [
        (
            "mm_ppu",
            6,
            {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "LOAD_MODE"},
        ),
        ("mm_ppu_small_m", 6, {"BLOCK_N", "BLOCK_K", "LOAD_MODE"}),
        (
            "mm_ppu_narrow_n",
            6,
            {"BLOCK_M", "BLOCK_N", "BLOCK_K", "LOAD_MODE"},
        ),
        (
            "mm_ppu_mid_m",
            7,
            {"BLOCK_M", "BLOCK_N", "BLOCK_K", "LOAD_MODE"},
        ),
        ("gemv_ppu", 5, {"BLOCK_M", "BLOCK_K"}),
        ("mm_ppu_multi_row_gemv", 5, {"BLOCK_M", "BLOCK_K"}),
        (
            "mm_ppu_grouped_row_gemv",
            5,
            {"BLOCK_M", "BLOCK_K", "ROWS_PER_PROGRAM"},
        ),
        (
            "mm_ppu_split_k",
            5,
            {
                "BLOCK_M",
                "BLOCK_N",
                "BLOCK_K",
                "SPLIT_K",
                "LOAD_MODE",
                "INTERLEAVED",
            },
        ),
        ("mm_ppu_split_k_reduce", 2, {"BLOCK", "VEC"}),
    ],
)
def test_ppu_mm_expand_spaces_are_registered_and_bounded(
    op_name, strategy_size, required_meta
):
    loader = TunedConfigLoader()
    expand = loader.get_expand_config(op_name, yaml_path=str(YAML_PATH))
    configs = loader.ops_get_configs(op_name, yaml_path=str(YAML_PATH))

    assert expand != -1
    assert expand["strategy"] == ["default"] * strategy_size
    assert 0 < len(configs) <= 5000
    assert all(required_meta <= config.kwargs.keys() for config in configs)


@pytest.mark.parametrize(
    ("op_name", "fixed_block_m"),
    [("mm_ppu", None), ("mm_ppu_small_m", 16), ("mm_ppu_mid_m", 32)],
)
def test_ppu_mm_expand_spaces_respect_resource_limits(
    op_name, fixed_block_m
):
    loader = TunedConfigLoader()
    configs = loader.ops_get_configs(op_name, yaml_path=str(YAML_PATH))

    for config in configs:
        assert config.kwargs["PIPE_STAGES"] == config.num_stages
        block_m = config.kwargs.get("BLOCK_M", fixed_block_m)
        shared_memory = (
            (block_m + config.kwargs["BLOCK_N"])
            * config.kwargs["BLOCK_K"]
            * 2
            * max(config.num_stages - 1, 1)
        )
        assert shared_memory <= 240 * 1024
        if op_name == "mm_ppu":
            accumulators_per_thread = (
                block_m
                * config.kwargs["BLOCK_N"]
                // (config.num_warps * 32)
            )
            assert accumulators_per_thread <= 128


@pytest.mark.parametrize(
    "op_name",
    [
        "mm_ppu_narrow_n",
        "gemv_ppu",
        "mm_ppu_multi_row_gemv",
        "mm_ppu_grouped_row_gemv",
        "mm_ppu_split_k",
    ],
)
def test_ppu_mm_pipeline_stage_metadata_is_coupled(op_name):
    loader = TunedConfigLoader()
    configs = loader.ops_get_configs(op_name, yaml_path=str(YAML_PATH))

    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages
        for config in configs
    )


def test_ppu_split_k_expand_space_covers_runtime_choices():
    loader = TunedConfigLoader()
    configs = loader.ops_get_configs(
        "mm_ppu_split_k", yaml_path=str(YAML_PATH)
    )

    assert {config.kwargs["SPLIT_K"] for config in configs} == {
        2,
        4,
        6,
        7,
        8,
    }
    assert {config.kwargs["INTERLEAVED"] for config in configs} == {
        False,
        True,
    }
