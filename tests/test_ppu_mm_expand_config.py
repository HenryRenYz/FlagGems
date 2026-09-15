# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

from flag_gems.runtime.configs_loader import TunedConfigLoader


def test_ppu_mm_expand_config_is_filtered_and_stage_coupled():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu", yaml_path=str(yaml_path))

    assert expand != -1
    assert expand["strategy"] == ["default"] * 6
    # Keep the expanded space bounded so a full PPU search remains practical;
    # the production shortlist is merged separately by LibTuner.
    assert 600 <= len(configs) <= 5000
    assert len(configs) <= 5000
    assert {config.kwargs["LOAD_MODE"] for config in configs} == {0, 1, 3}
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages for config in configs
    )
    assert all(
        config.kwargs["BLOCK_M"]
        * config.kwargs["BLOCK_N"]
        // (config.num_warps * 32)
        <= 128
        for config in configs
    )
    assert all(
        (
            config.kwargs["BLOCK_M"] + config.kwargs["BLOCK_N"]
        )
        * config.kwargs["BLOCK_K"]
        * 2
        * max(config.num_stages - 1, 1)
        <= 240 * 1024
        for config in configs
    )


def test_ppu_small_m_expand_config_keeps_a_legal_fixed_m_tile():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu_small_m", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu_small_m", yaml_path=str(yaml_path))

    assert expand != -1
    assert expand["strategy"] == ["default"] * 6
    assert len(configs) == 280
    assert {config.kwargs["LOAD_MODE"] for config in configs} == {0, 1, 2, 3}
    assert all("BLOCK_M" not in config.kwargs for config in configs)
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages for config in configs
    )
    assert all(
        (16 + config.kwargs["BLOCK_N"])
        * config.kwargs["BLOCK_K"]
        * 2
        * max(config.num_stages - 1, 1)
        <= 240 * 1024
        for config in configs
    )


def test_ppu_grouped_row_gemv_expand_config_is_registered():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config(
        "mm_ppu_grouped_row_gemv", yaml_path=str(yaml_path)
    )
    configs = loader.ops_get_configs(
        "mm_ppu_grouped_row_gemv", yaml_path=str(yaml_path)
    )

    assert expand != -1
    assert expand["strategy"] == ["default"] * 5
    assert len(configs) == 36
    assert {config.kwargs["BLOCK_M"] for config in configs} == {16, 32}
    assert {config.kwargs["ROWS_PER_PROGRAM"] for config in configs} == {16}


def test_ppu_narrow_n_expand_config_retains_measured_stage_five_tiles():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu_narrow_n", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu_narrow_n", yaml_path=str(yaml_path))

    assert expand["strategy"] == ["default"] * 6
    assert len(configs) == 1296
    assert {config.kwargs["BLOCK_M"] for config in configs} == {
        16,
        32,
        64,
        128,
        256,
        512,
    }
    assert {config.kwargs["BLOCK_N"] for config in configs} == {16, 32, 64}
    assert {config.kwargs["BLOCK_K"] for config in configs} == {32, 64, 128}
    assert {config.num_stages for config in configs} == {2, 3, 4, 5}
    assert {config.num_warps for config in configs} == {4, 8}
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages for config in configs
    )


def test_ppu_mid_m_expand_config_has_a_stable_cache_schema():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu_mid_m", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu_mid_m", yaml_path=str(yaml_path))

    assert expand != -1
    assert expand["strategy"] == ["default"] * 7
    assert len(configs) == 248
    assert {config.kwargs["BLOCK_M"] for config in configs} == {32}
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages for config in configs
    )
    assert all(
        (32 + config.kwargs["BLOCK_N"])
        * config.kwargs["BLOCK_K"]
        * 2
        * max(config.num_stages - 1, 1)
        <= 240 * 1024
        for config in configs
    )


def test_ppu_gemv_expand_configs_are_registered():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    gemv_expand = loader.get_expand_config(
        "gemv_ppu", yaml_path=str(yaml_path)
    )
    gemv_configs = loader.ops_get_configs(
        "gemv_ppu", yaml_path=str(yaml_path)
    )
    assert gemv_expand["strategy"] == ["default"] * 5
    assert len(gemv_configs) == 294
    assert all(
        {"BLOCK_M", "BLOCK_K"} <= config.kwargs.keys()
        for config in gemv_configs
    )
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages
        for config in gemv_configs
    )

    multi_row_expand = loader.get_expand_config(
        "mm_ppu_multi_row_gemv", yaml_path=str(yaml_path)
    )
    multi_row_configs = loader.ops_get_configs(
        "mm_ppu_multi_row_gemv", yaml_path=str(yaml_path)
    )
    assert multi_row_expand["strategy"] == ["default"] * 5
    assert len(multi_row_configs) == 100
    assert all(
        {"BLOCK_M", "BLOCK_K", "PIPE_STAGES"} <= config.kwargs.keys()
        for config in multi_row_configs
    )
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages
        for config in multi_row_configs
    )


def test_ppu_split_k_expand_configs_are_bounded_and_complete():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu_split_k", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu_split_k", yaml_path=str(yaml_path))
    assert expand["strategy"] == ["default"] * 5
    assert len(configs) == 4320
    assert len(configs) <= 5000
    assert {config.kwargs["SPLIT_K"] for config in configs} == {2, 4, 6, 7, 8}
    assert all("INTERLEAVED" in config.kwargs for config in configs)
