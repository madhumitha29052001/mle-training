import os

from HousePricePrediction import train


def test_arguments_train():
    args = train.parse_args()
    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.pkl_path)
    assert os.path.exists(args.log_path)


def test_model_files():
    train.main()
    assert os.path.exists("../../artifacts/linear_regression.pkl")
    assert os.path.exists("../../artifacts/decision_tree.pkl")
    assert os.path.exists("../../artifacts/final_model.pkl")
    assert os.path.exists("../../artifacts/random_forest_random_search.pkl")
