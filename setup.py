from pathlib import Path

from setuptools import setup  # , find_packages

this_directory = Path(__file__).parent

long_description = (this_directory / "README.md").read_text()


setup(
    name="HousePricePrediction",
    version="0.3",
    author="Madhumitha R",
    author_email="madhumitha.ravik@tigeranalytics.com",
    packages=["HousePricePrediction"],
    package_dir={"HousePricePrediction": "src/HousePricePrediction"},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
