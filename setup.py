import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LearnRL",
    version="0.0.1",
    author="Mathïs Fédérico, Théo Combey",
    author_email="mathfederico@gmail.com",
    description="A package to learn about Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MathisFederico/LearnRL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: All rights reserved",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)