import argparse
import os
import tarfile

import numpy as np
import pandas as pd

# from configure_logging import configure_logger
from six.moves import urllib  # pyright:ignore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

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
        "--download_dir",
        type=str,
        default="/mnt/d/mle-training/data/raw",
        help="Where to download the data",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/d/mle-training/data/processed",
        help="Path to save the transformed data",
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
        default="../../logs/ingest_data.log",
        help="Path of the log file to be saved",
    )

    parser.add_argument(
        "--no_console_log",
        action="store_false",
        help="include this to not print logs in console",
    )

    return parser.parse_args()


def fetch_housing_data(housing_url, housing_path):
    """Download the dataset from URL and save in the given path

    Parameters
    ----------
    housing_url : str
        The URL to the dataset
    housing_path : str
        Path to save the downloaded dataset
    """
    if os.path.exists(os.path.join(housing_path, "housing.tgz")):
        return
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """Load the dataset as Pandas dataframe and return

    Parameters
    ----------
    housing_path : str
        Path to the csv file

    Returns
    -------
    Pandas Dataframe
        The loaded dataframe object
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def transform_data(data_folder, output_folder, args):
    """Transform the raw data into train and test and save them

    Parameters
    ----------
    data_folder : _str
        Path to the raw data
    output_folder : str
        Path to save the processed data
    args : str
        command line arguments

    Returns
    -------
    None
    """
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = data_folder  # os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    logger = configure_logger(
        log_file=args.log_path, console=args.no_console_log, log_level=args.log_level
    )
    logger.info("Downloading data...")
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    logger.info("loading data...")
    housing = load_housing_data(HOUSING_PATH)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    logger.debug("Splitting the data using StratifiedShuffleSplit")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)

    logger.debug("Splitting using train_test_split")
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    logger.info("Calculating errors of splits")
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    logger.debug("Feature elimination using correlation")
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    logger.debug("Imputing")
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    logger.info("Saving train data....")
    housing_prepared.to_csv(os.path.join(output_folder, "X_train.csv"), index=False)
    housing_labels.to_csv(os.path.join(output_folder, "Y_train.csv"), index=False)

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    logger.info("Saving test data....")
    X_test_prepared.to_csv(os.path.join(output_folder, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_folder, "Y_test.csv"), index=False)


def main():
    """To call transform data function with the command line args"""
    args = parse_args()
    transform_data(data_folder=args.download_dir, output_folder=args.out_dir, args=args)


if __name__ == "__main__":
    main()
