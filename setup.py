# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LearnRL",
    version="0.1.0",
    author="Mathïs Fédérico",
    author_email="mathfederico@gmail.com",
    description="A package to learn about Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MathisFederico/LearnRL",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'gym',
        'pygame'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU LPGLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)