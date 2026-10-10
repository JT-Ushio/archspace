from _qk_common import make_build_config

build_config = make_build_config("f2qk")

if __name__ == "__main__":
    from olmo_core.script_utils import main
    main(build_config)
