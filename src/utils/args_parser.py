
import argparse



class ArgParser:

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(
            description="Full pipeline: preprocess → train → cross‐validate → evaluate"
        )
        parser.add_argument(
            "-c",
            "--public-config",
            dest="public_config",
            help="Path to public YAML config file (no secrets).",
            default="config/public/config.yaml",
        )
        parser.add_argument(
            "-s",
            "--private-config",
            dest="private_config",
            help="Path to private YAML config file (with secrets).",
            default="config/private/config.yaml",
        )
        return parser.parse_args()