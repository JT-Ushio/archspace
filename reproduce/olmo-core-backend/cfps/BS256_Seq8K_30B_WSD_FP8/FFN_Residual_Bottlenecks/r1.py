"""R1: unshared FFN, width 2048, rank 1280, wide=true, latent=false."""

from _common import make_build_config

build_config = make_build_config("r1")

if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
