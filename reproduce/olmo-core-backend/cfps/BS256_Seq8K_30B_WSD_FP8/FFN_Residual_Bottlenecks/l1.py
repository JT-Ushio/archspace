"""L1: unshared FFN, width 8192, rank 512, wide=false, latent=true."""

from _common import make_build_config

build_config = make_build_config("l1")

if __name__ == "__main__":
    from olmo_core.script_utils import main

    main(build_config)
