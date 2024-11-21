# GeMax: Learning Graph Representation via Graph Entropy Maximization
import argparse
import yaml
from utils.utils import set_random_seed
from experiment import experiment_gemax
from evaluate import evaluate_gemax


def main():
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="GeMax for Graph Representation Learning")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_random_seed(config["seed"])

    if config["mode"] == "train":
        experiment_gemax(config)
    elif config["mode"] == "evaluate":
        evaluate_gemax(config)
    else:
        raise ValueError(f"Invalid mode: {config['mode']}")

if __name__ == "__main__":
    main()