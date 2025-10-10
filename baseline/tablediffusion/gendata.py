import argparse
import json
from pathlib import Path
import pandas as pd

from models import TableDiffusion_Synthesiser


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data using TableDiffusion")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (nursery, titanic, adult, bank_marketing, wine_quality)")
    parser.add_argument("--epsilon", type=str, required=True, help="Privacy budget(s)")
    parser.add_argument("--delta", type=str, default="auto", help="Privacy parameter delta")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs for TableDiffusion")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for TableDiffusion")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for TableDiffusion")
    parser.add_argument("--diffusion_steps", type=int, default=3, help="Number of diffusion steps for TableDiffusion")
    parser.add_argument("--n_instances", type=int, default=None, help="Sample only n instances from training data")
    return parser.parse_args()

def get_delta(args, n):
    if args.delta == 'auto':
        return 1/n
    return float(args.delta)


def generate_synthetic_data(train_df, n, args):
    synthesiser = TableDiffusion_Synthesiser(
        batch_size=args.batch_size,
        lr=args.lr,
        diffusion_steps=args.diffusion_steps,
        predict_noise=True,
        epsilon_target=args.epsilon,
        epoch_target=args.epoch * args.diffusion_steps,
        mlflow_logging=False,
        cuda=True
    )

    synthesiser.fit(train_df, n_epochs=args.epoch, epsilon=args.epsilon, discrete_columns=[], verbose=True)
    synthetic_data = synthesiser.sample(n=n, post_process=True)
    return synthetic_data


def main():
    args = parse_args()

    output_dir = f"data/{args.dataset}/dpsyn/eps{args.epsilon}/table_diffusion"
    args.epsilon = float(args.epsilon)

    valid_datasets = ['nursery', 'titanic', 'adult', 'bank_marketing', 'wine_quality']
    if args.dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset type: {args.dataset}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading configuration and data...")
    config_path = f"src/dataset/{args.dataset}/{args.dataset}.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_df = pd.read_csv(f"data/{args.dataset}/clean/{args.dataset}_train_onehot.csv")

    if args.n_instances is not None:
        train_df = train_df.sample(n=args.n_instances)

    args.delta = get_delta(args, len(train_df))
    print(f"Dataset size: {len(train_df)}")
    print(f"Using epsilon = {args.epsilon}, delta = {args.delta}")

    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data(train_df, len(train_df), args)

    output_path = f"{output_dir}/synthetic_data.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")


if __name__ == "__main__":
    main()
