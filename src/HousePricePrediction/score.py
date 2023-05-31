import argparse
import logging
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from configure_logging import configure_logger

# from configure_logging import configure_logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from HousePricePrediction.configure_logging import configure_logger


def parse_args():
    """Function to parse the arguments

    Returns
    -------
    Args
        Returns the arguments that are added in the argument parser

    """
    parser = argparse.ArgumentParser(
        description="Script to download and transform data"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/d/mle-training/artifacts",
        help="Path to the models",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/processed",
        help="Path to the transformed data",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Specify the log level",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="../../logs/score.log",
        help="Path of the log file to be saved",
    )

    parser.add_argument(
        "--no_console_log",
        action="store_false",
        help="include this to not print logs in console",
    )

    return parser.parse_args()


def predict_score(processed_path, model_path):
    """This function returns the scores of the given model

    Parameters
    ----------
    processed_path : str
        Path to the processed test data
    model_path : str
        Path to the ML model pickle
    """

    logging.info("Downloading the Xtest and Ytest from processed path folder")
    # Reading the datasets
    X_test_prepared = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_path, "Y_test.csv"))

    os.chdir(model_path)

    # load the model from disk
    for file in os.listdir(model_path):
        if file == ".gitkeep":
            pass
        else:
            loaded_model = pickle.load(open(file, "rb"))
            final_predictions = loaded_model.predict(X_test_prepared)
            final_mse = mean_squared_error(y_test, final_predictions)
            final_rmse = np.sqrt(final_mse)
            r2 = r2_score(y_test, final_predictions)
            mae = mean_absolute_error(y_test, final_predictions)
            logging.info(f"{loaded_model} metrics:")
            logging.info(f"    Mean Squared Error: {final_mse}")
            logging.info(f"    Root Mean Squared Error: {final_rmse}")
            logging.info(f"    r2 score: {r2}")
            logging.info(f"    Mean absolute error: {mae}")

            mlflow.log_metric("mse", final_mse)
            mlflow.log_metric("rmse", final_rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


def main():
    args = parse_args()

    model_path = args.model_dir
    processed_path = args.data_dir
    log_file_path = args.log_path
    configure_logger(
        log_file=log_file_path, console=args.no_console_log, log_level=args.log_level
    )
    # print(model_path)
    predict_score(processed_path, model_path)


if __name__ == "__main__":
    main()
