import os

from HousePricePrediction import score


def test_arguments():
    args = score.parse_args()
    assert os.path.exists(args.model_dir)
    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.log_path)
