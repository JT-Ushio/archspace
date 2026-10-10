"""P4: shared rank160, per-projection private rank368."""
from _common import make_build_config

build_config = make_build_config("p4")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
