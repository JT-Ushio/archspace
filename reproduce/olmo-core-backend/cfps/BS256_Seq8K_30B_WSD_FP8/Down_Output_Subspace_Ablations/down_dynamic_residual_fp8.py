"""Down output-subspace ablation: dynamic rotation with a zero-start residual gate."""
from olmo_core.script_utils import main
from _common import build_rotation_config


def build_config(opts, overrides):
    return build_rotation_config(opts, overrides, "dynamic_residual")


if __name__ == "__main__":
    main(build_config)
