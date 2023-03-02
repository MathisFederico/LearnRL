


import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "pypi-readme.rst").read_text()

def get_version():
    version_file = open('VERSION')
    return version_file.read().strip()
VERSION = get_version()

def get_requirements():
    requirements_file = open('requirements.txt')
    return requirements_file.readlines()
REQUIREMENTS = get_requirements()

setup(
    name="learnrl",
    version=VERSION,
    author="Mathïs Fédérico",
    author_email="mathfederico@gmail.com",
    description="A package to learn about Reinforcement Learning",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/MathisFederico/LearnRL",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)