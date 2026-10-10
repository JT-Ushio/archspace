"""P4 partial sharing with Q/K SiLU, keeping the exact rank512 budget."""
from _qk_common import make_build_config

build_config = make_build_config("p4qk")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
