"""F2: UG256, Down1024, UG sharing=False."""
from _common import make_build_config

build_config = make_build_config("f2")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
