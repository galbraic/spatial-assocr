from setuptools import setup, find_packages

setup(
    name="spatial-assocr",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "sklearn", "pandas", "plotly", "geopy"],
)
