import argparse


class ArgParser:

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(
            description="Full pipeline: preprocess → train → cross‐validate → evaluate"
        )
        parser.add_argument(
            "--config-dir",
            dest="config_dir",
            help="Path to the config directory containing public/ and private/ subdirectories.",
            default="config",
        )
        return parser.parse_args()
