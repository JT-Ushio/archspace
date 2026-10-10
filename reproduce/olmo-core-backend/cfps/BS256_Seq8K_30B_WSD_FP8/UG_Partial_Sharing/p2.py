"""P2: shared rank480, per-projection private rank80."""
from _common import make_build_config

build_config = make_build_config("p2")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
