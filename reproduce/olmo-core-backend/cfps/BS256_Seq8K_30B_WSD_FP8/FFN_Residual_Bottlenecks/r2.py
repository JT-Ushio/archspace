"""R2: unshared FFN, width 2048, rank 1280, wide=false, latent=true."""

from _common import make_build_config

build_config = make_build_config("r2")

if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
