from setuptools import setup, find_packages

setup(
    name="spatial-assocr",
    packages=find_packages(),
    dependency_links=["git+https://github.com/klicperajo/pyemd.git#egg=pyemd"],
    install_requires=["numpy", "scipy", "pandas", "plotly", "pyemd"],
)
