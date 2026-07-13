"""
PharmaGuard
Main Entry Point
"""

import argparse


def main():

    parser = argparse.ArgumentParser(
        description="PharmaGuard Machine Learning Pipeline"
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "eda",
            "preprocess",
            "train",
            "evaluate",
            "compare",
            "predict"
        ],
        help="Select the task to execute."
    )

    parser.add_argument(
        "--model",
        required=False,
        choices=[
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "elliptic_envelope",
            "autoencoder"
        ],
        help="Model name (required only for training)."
    )

    args = parser.parse_args()

    print(f"\nTask Selected : {args.task}")

    if args.task == "eda":

        print("Launching Exploratory Data Analysis...\n")

        from ml.eda.eda import main as eda_main

        eda_main()

    elif args.task == "preprocess":

        print("Launching Preprocessing...\n")

        from ml.preprocessing.preprocessing import main as preprocess_main

        preprocess_main()

    elif args.task == "train":

        if args.model is None:
            parser.error("--model is required when using --task train")

        if args.model == "isolation_forest":

            from ml.models.train_isolation_forest import (
                main as train_if
            )

            train_if()

        elif args.model == "local_outlier_factor":

            from ml.models.train_local_outlier_factor import (
                main as train_lof
            )

            train_lof()

        elif args.model == "one_class_svm":

            from ml.models.train_one_class_svm import (
                main as train_ocsvm
            )

            train_ocsvm()

        elif args.model == "autoencoder":

            from ml.models.train_autoencoder import (
                main as train_autoencoder
            )

            train_autoencoder()
        

        else:

            print(f"{args.model} coming soon.")

    elif args.task == "evaluate":

        print("Evaluation Module Coming Soon.")

    elif args.task == "compare":

        print("Model Comparison Module Coming Soon.")

    elif args.task == "predict":

        print("Prediction Module Coming Soon.")


if __name__ == "__main__":
    main()