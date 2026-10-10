"""P1: shared rank560, per-projection private rank0."""
from _common import make_build_config

build_config = make_build_config("p1")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
