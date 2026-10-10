"""P3: shared rank320, per-projection private rank224."""
from _common import make_build_config

build_config = make_build_config("p3")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
