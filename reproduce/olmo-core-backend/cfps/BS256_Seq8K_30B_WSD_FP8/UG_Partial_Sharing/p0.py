"""P0: shared rank0, per-projection private rank512."""
from _common import make_build_config

build_config = make_build_config("p0")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
