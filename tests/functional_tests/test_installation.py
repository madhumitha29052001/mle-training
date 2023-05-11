def test_pkg_installation():
    try:
        import HousePricePrediction
    except Exception as e:
        assert (
            False
        ), f"Error : {e}. HousePricePrediction pacakage is not installed correctly"

    try:
        import numpy
        import pandas
    except Exception as e:
        assert False, f"Error : {e}. Numpy/Pandas pacakage is not installed correctly"
