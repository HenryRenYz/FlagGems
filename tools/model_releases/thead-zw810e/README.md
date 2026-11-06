# T-Head PPU model release 1.2.0

The release contains 35 retrained GEMM models and the two retained MUL models.
Training uses a fixed random half of the 2,372 original shapes, with 1,186 shapes
held out and seed 2026. Layouts, batch sizes and public operators share the same
split. The GEMM coverage includes MM, ADDMM, router_gemm, BMM and BADDBMM in BF16.

The trained package previously labelled 1.3.0 is now release 1.2.0. Every child
model has version 1.2.0, with updated config digests in its XGBoost binding and
training summary. The trees, feature definitions and measured performance data
are unchanged. The previous unpublished 1.2.0 draft is withdrawn. Its checksum
is recorded in [1.2.0.json](1.2.0.json), together with the checksum of this release.

The runtime publication manifest continues to advertise 1.1.0 until this archive
is uploaded. At publication, add the uploaded HTTPS URL and the new checksum
under `thead-zw810e.versions.1.2.0` in the runtime manifest.

## Use the release

Activate the SDK 2.1 environment, then point FlagTune at the package directory:

```bash
export FLAGTUNE_MODEL_DIR=/path/to/release-1.2.0/package
export FLAGTUNE_MODEL_VERSION=1.2.0
export FLAGTUNE_DISABLE_REMOTE=1
unset USE_FLAGTUNE
export USE_FLAGTUNE_COST_MODEL=1 FLAGTUNE_INCLUDE=mm,bmm,mul
export FLAGTUNE_RUNTIME_CACHE="$HOME/.cache/flagtune/1.2.0-a2764fb4319d"
export FLAGTUNE_MODEL_CACHE="$FLAGTUNE_RUNTIME_CACHE/models"
export FLAGGEMS_CACHE_DIR="$FLAGTUNE_RUNTIME_CACHE/flaggems"
export FLAGGEMS_DB_URL="sqlite:///$FLAGTUNE_RUNTIME_CACHE/benchmark.db"
export TRITON_CACHE_DIR="$FLAGTUNE_RUNTIME_CACHE/triton"
mkdir -p "$FLAGTUNE_MODEL_CACHE" "$FLAGGEMS_CACHE_DIR" "$TRITON_CACHE_DIR"
```

The cache directory includes the release checksum to distinguish it from the
withdrawn draft, which used the same version number.

## Reproduce the version change

Run from the FlagGems checkout with FlagTree and PyYAML installed:

```bash
python tools/reversion_model_package.py \
  --source /path/to/draft/thead-zw810e_v1.3.0.tar.gz \
  --source-version 1.3.0 \
  --version 1.2.0 \
  --output /path/to/release-1.2.0
```

The tool writes a new directory and verifies that all model data except the
config binding remains identical to the source package.
