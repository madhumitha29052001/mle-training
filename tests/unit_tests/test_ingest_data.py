import os

import pandas as pd

from HousePricePrediction import ingest_data


def test_arguments():
    args = ingest_data.parse_args()
    assert os.path.exists(args.download_dir)
    assert os.path.exists(args.out_dir)
    assert os.path.exists(args.log_path)


def test_fetch_dataset():
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "data/raw"
    print("housing_path :", HOUSING_PATH)
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    ingest_data.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    assert os.path.exists("../../data/raw/housing.csv")


def test_load_dataset():
    raw_data_path = "data/raw/"
    df = ingest_data.load_housing_data(raw_data_path)
    assert isinstance(df, pd.DataFrame)


def test_transform_data():
    args = ingest_data.parse_args()
    ingest_data.transform_data(args.download_dir, args.out_dir, args)
    assert os.path.exists("../../data/processed/X_train.csv")
    assert os.path.exists("../../data/processed/Y_train.csv")
    assert os.path.exists("../../data/processed/X_test.csv")
    assert os.path.exists("../../data/processed/Y_test.csv")
