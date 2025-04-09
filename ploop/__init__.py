"""
P-LOOP: Parallelizable Bayesian Optimisation Framework

This package provides tools for running Bayesian optimisation, particularly
focused on parallel execution of objective function evaluations.
"""

import logging

# --- Core Classes for User Interaction ---

# Import the main controller class users will likely instantiate.
from .controllers import GaussianProcessController

# Import the base Controller class in case users want to extend it.
from .controllers import Controller

# Import the WorkerInterface, as users MUST subclass this to define their experiment/objective function.
from .interfaces import WorkerInterface


# --- Public API Definition ---

# Define __all__ to specify which symbols are imported when using 'from ploop import *'
# and to clearly define the public interface of the package.
__all__ = [
    "GaussianProcessController",
    "Controller",
    "WorkerInterface",
]


# --- Package Metadata (Optional) ---

# You can define version information here or load it from another file
__version__ = "0.1.0"  # Example version


# --- Logging Configuration ---

# Add a NullHandler to the package's root logger.
# This prevents log messages from propagating to the root logger if the
# user application doesn't configure logging. It's standard practice for libraries.
logging.getLogger(__name__).addHandler(logging.NullHandler())
