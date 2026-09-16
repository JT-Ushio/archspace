"""Down output-subspace ablation: dynamic."""
from olmo_core.script_utils import main
from _common import build_rotation_config


def build_config(opts, overrides):
    return build_rotation_config(opts, overrides, "dynamic")


if __name__ == "__main__":
    main(build_config)
