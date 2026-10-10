"""Q/K SiLU interaction with the primary UG-sharing recipe."""
from _qk_common import make_build_config

build_config = make_build_config("p0qk")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
