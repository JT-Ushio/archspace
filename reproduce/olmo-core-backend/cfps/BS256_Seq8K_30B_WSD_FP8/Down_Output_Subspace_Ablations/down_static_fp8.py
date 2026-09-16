"""Down output-subspace ablation: static."""
from olmo_core.script_utils import main
from _common import build_rotation_config


def build_config(opts, overrides):
    return build_rotation_config(opts, overrides, "static")


if __name__ == "__main__":
    main(build_config)
