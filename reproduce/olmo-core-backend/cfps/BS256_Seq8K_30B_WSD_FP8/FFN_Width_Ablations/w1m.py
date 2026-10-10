"""W1-M: reduce C2's FFN width only, without reallocating saved parameters."""

from _common import make_build_config

build_config = make_build_config("w1m")

if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
