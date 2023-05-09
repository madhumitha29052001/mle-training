import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd

# from configure_logging import configure_logger
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from HousePricePrediction.configure_logging import configure_logger


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
        "--data_dir",
        type=str,
        default="../../data/processed",
        help="Path to the prepared data",
    )

    parser.add_argument(
        "--pkl_path",
        type=str,
        default="../../artifacts",
        help="Path to the model pickles",
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
        default="../../logs/train.log",
        help="Path of the log file to be saved",
    )

    parser.add_argument(
        "--no_console_log",
        action="store_false",
        help="include this to not print logs in console",
    )

    return parser.parse_args()


def train_models(housing_prepared, housing_labels, pkl_path):
    """This function trains various Machine learning models and save the model weights in pickle format

    Parameters
    ----------
    housing_prepared : Pandas dataframe
        The processed train data
    housing_labels : Pandas dataframe
        Labels for train data
    pkl_path : str
        Path to save the pickle files
    """
    logging.info("Building Linear Regression model")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    logging.info("Saving the Linear Regression model")
    filename = os.path.join(pkl_path, "linear_regression.pkl")
    pickle.dump(lin_reg, open(filename, "wb"))

    logging.info("Building DecisionTreeRegressor model")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    logging.info("Saving the DecisionTreeRegressor model")
    filename = os.path.join(pkl_path, "decision_tree.pkl")
    pickle.dump(tree_reg, open(filename, "wb"))

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels.values.ravel())
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print("RandomSearch_RandomForest_Model", np.sqrt(-mean_score), params)

    rnd_search_best_estimator = rnd_search.best_estimator_
    logging.info("Saving the RandomSearch_RandomForest_Model")
    filename = os.path.join(pkl_path, "random_forest_random_search.pkl")
    pickle.dump(rnd_search_best_estimator, open(filename, "wb"))

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels.values.ravel())

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    logging.info("Final model obtained using grid_search.best_estimator_")
    final_model = grid_search.best_estimator_

    logging.info("Saving the final model")
    filename = os.path.join(pkl_path, "final_model.pkl")
    pickle.dump(final_model, open(filename, "wb"))


def main():
    """Main function to parse arguments and train models"""
    args = parse_args()
    configure_logger(
        log_file=args.log_path, console=args.no_console_log, log_level=args.log_level
    )
    housing_prepared = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    housing_labels = pd.read_csv(os.path.join(args.data_dir, "Y_train.csv"))

    train_models(housing_prepared, housing_labels, args.pkl_path)


if __name__ == "__main__":
    main()
