from olmo_core.script_utils import main
from _common import build_init_config


def build_config(opts, overrides):
    return build_init_config(opts, overrides, "independent")


if __name__ == "__main__":
    main(build_config)
