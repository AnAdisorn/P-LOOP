import setuptools
import os
import re


# Function to extract the version string from __init__.py
def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py = open(os.path.join(package, "__init__.py")).read()
    # Regex adjusted to match the specific format in your __init__.py
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', init_py, re.MULTILINE)
    if match:
        return match.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {package}/__init__.py.")


# Function to read the long description from README.md
def get_long_description(readme_path="README.md"):
    """
    Return the README.md file's contents.
    """
    try:
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        print(f"Warning: Could not find {readme_path} to use as long description.")
        return ""  # Return empty string if README is not found


# --- Package Configuration ---
PACKAGE_NAME = "ploop"
VERSION = get_version(PACKAGE_NAME)
AUTHOR = "Adisorn Panasawatwong"
AUTHOR_EMAIL = "adisornbkk@gmail.com"
DESCRIPTION = "P-LOOP: Parallelizable Bayesian Optimisation Framework"
LONG_DESCRIPTION = get_long_description()
URL = "https://github.com/AnAdisorn/P-LOOP"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.8"  # Based on f-strings and common library support

# --- Dependencies ---
# Scan your .py files for imports to determine these
INSTALL_REQUIRES = [
    "numpy>=1.20",  # Widely used for numerical operations
    "scipy>=1.7",  # Used in learners (optimize, stats) and utils (stats)
    "scikit-learn>=1.0",  # Used for GaussianProcessRegressor, kernels, scalers
    # Standard library modules (like logging, multiprocessing, queue, threading, pathlib, datetime, os, re, time, subprocess)
    # do NOT need to be listed here.
]

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",  # Tells PyPI to render README as Markdown
    url=URL,
    license=LICENSE,
    packages=setuptools.find_packages(
        exclude=("tests*",)
    ),
    # include_package_data=True,
    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     "": ["*.txt", "*.rst"],
    #     # And include any *.dat files found in the 'data' subdirectory
    #     # of the 'mypkg' package, also:
    #     "ploop": ["data/*.dat"],
    # },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=PYTHON_REQUIRES,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        f"License :: OSI Approved :: {LICENSE} License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="bayesian optimisation, optimization, machine learning, gaussian process, parallel, asynchronous",
)
