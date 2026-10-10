"""G1: C2-based experiment; see scripts/train/FFN_Width_Ablations/README.md."""

from _common import make_build_config

build_config = make_build_config("g1")

if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
