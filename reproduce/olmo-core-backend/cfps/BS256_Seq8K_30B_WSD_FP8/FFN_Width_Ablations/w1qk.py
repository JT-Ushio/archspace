"""W1-QK: W1 with Q SiLU; all shapes and training settings stay identical."""

from _common import make_build_config

build_config = make_build_config("w1qk")

if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
