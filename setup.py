from setuptools import find_packages
from setuptools import setup

setup(
    name="causal_forest",
    version="0.0.1",
    description="Algorithm to estimate heterogeneous treatment effects.",
    long_description="""The package causal_forest implements the causal
    forest algorithm from Athey and Wager (2018), which can be used to
    estimate heterogeneous treatment effects.""",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 1 - Planning",
    ],
    keywords=["econometrics", "causal inference", "machine learning"],
    url="https://github.com/timmens/causal-forest",
    author="Tim Mensinger",
    author_email="tim.mensinger@uni-bonn.de",
    packages=find_packages(exclude=["tests/*"]),
    zip_safe=False,
    # package_data={"estimagic": ["optimization/algo_dict.json"]},
    # include_package_data=True,
)
