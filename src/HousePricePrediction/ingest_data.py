import argparse
import os
import re
import tarfile

import mlflow
import numpy as np
import pandas as pd
from configure_logging import configure_logger
from six.moves import urllib  # pyright:ignore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        "--download_dir",
        type=str,
        default="../../data/raw",
        help="Where to download the data",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="../../data/processed",
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


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def columns(self):
        if self.add_bedrooms_per_room:
            cols = [
                "rooms_per_household",
                "population_per_household",
                "bedrooms_per_room",
            ]
        else:
            cols = ["rooms_per_household", "population_per_household"]
        return cols


def get_feature_names_from_column_transformer(col_trans):
    """Get feature names from a sklearn column transformer.

    The `ColumnTransformer` class in `scikit-learn` supports taking in a
    `pd.DataFrame` object and specifying `Transformer` operations on columns.
    The output of the `ColumnTransformer` is a numpy array that can used and
    does not contain the column names from the original dataframe. The class
    provides a `get_feature_names` method for this purpose that returns the
    column names corr. to the output array. Unfortunately, not all
    `scikit-learn` classes provide this method (e.g. `Pipeline`) and still
    being actively worked upon.

        NOTE: This utility function is a temporary solution until the proper fix is
    available in the `scikit-learn` library.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder as skohe

    # SimpleImputer has `add_indicator` attribute that distinguishes it from other transformers
    # Encoder had `get_feature_names` attribute that distinguishes it from other transformers
    # The last transformer is ColumnTransformer's 'remainder'
    col_name = []
    for transformer_in_columns in col_trans.transformers_:
        is_pipeline = 0
        raw_col_name = list(transformer_in_columns[2])

        if isinstance(transformer_in_columns[1], Pipeline):
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
            is_pipeline = 1
        else:
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, str):
                if transformer == "passthrough":
                    names = transformer._feature_names_in[raw_col_name].tolist()

                elif transformer == "drop":
                    names = []

                else:
                    raise RuntimeError(
                        f"Unexpected transformer action for unaccounted cols :"
                        f"{transformer} : {raw_col_name}"
                    )

            elif isinstance(transformer, skohe):
                names = list(transformer.get_feature_names(raw_col_name))

            elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [
                    raw_col_name[idx] + "_missing_flag"
                    for idx in missing_indicator_indices
                ]

                names = raw_col_name + missing_indicators

            else:
                names = list(transformer.get_feature_names())

        except AttributeError as error:
            names = raw_col_name
        if is_pipeline:
            names = [f"{transformer_in_columns[0]}_{col_}" for col_ in names]
        col_name.extend(names)

    return col_name


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
    housing_num = housing.drop("ocean_proximity", axis=1)

    attr_adder = CombinedAttributesAdder()
    cols = attr_adder.columns()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    housing_prepared_numpyarray = full_pipeline.fit_transform(housing)

    column_names = get_feature_names_from_column_transformer(full_pipeline)

    house_prep = (
        pd.DataFrame(housing_prepared_numpyarray[:, :8], columns=column_names[:8])
    ).join(
        (pd.DataFrame(housing_prepared_numpyarray[:, 8:11], columns=cols)).join(
            pd.DataFrame(housing_prepared_numpyarray[:, 11:], columns=column_names[8:])
        )
    )

    for i in range(len(house_prep.columns)):
        if "num" in house_prep.columns[i]:
            house_prep.rename(
                columns={
                    house_prep.columns[i]: re.sub("num_", "", house_prep.columns[i])
                },
                inplace=True,
            )

    housing_prepared = house_prep

    logger.info("Saving train data....")

    housing_prepared.to_csv(os.path.join(output_folder, "X_train.csv"), index=False)
    housing_labels.to_csv(os.path.join(output_folder, "Y_train.csv"), index=False)

    mlflow.log_artifact(os.path.join(output_folder, "X_train.csv"))
    mlflow.log_artifact(os.path.join(output_folder, "Y_train.csv"))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    data = full_pipeline.fit_transform(X_test)

    l1 = get_feature_names_from_column_transformer(full_pipeline)
    X_prep = (
        pd.DataFrame(data[:, :8], columns=l1[:8])
        .join(pd.DataFrame(data[:, 8:11], columns=cols))
        .join(pd.DataFrame(data[:, 11:], columns=l1[8:]))
    )

    for i in range(len(X_prep.columns)):
        if "num" in X_prep.columns[i]:
            X_prep.rename(
                columns={X_prep.columns[i]: re.sub("num_", "", X_prep.columns[i])},
                inplace=True,
            )

    X_test_prepared = X_prep
    logger.debug("Performed all the preprocessing on test data.")
    logger.debug("Downloading the test data")
    logger.info("Saving test data....")

    X_test_prepared.to_csv(os.path.join(output_folder, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_folder, "Y_test.csv"), index=False)

    mlflow.log_artifact(os.path.join(output_folder, "X_test.csv"))
    mlflow.log_artifact(os.path.join(output_folder, "Y_test.csv"))


def main():
    """To call transform data function with the command line args"""
    args = parse_args()
    transform_data(data_folder=args.download_dir, output_folder=args.out_dir, args=args)


if __name__ == "__main__":
    main()
