# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.rst").read_text()

setup(
    name="learnrl",
    version="0.1.1",
    author="Mathïs Fédérico",
    author_email="mathfederico@gmail.com",
    description="A package to learn about Reinforcement Learning",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/MathisFederico/LearnRL",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=[
        'numpy',
        'gym',
        'pygame'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)