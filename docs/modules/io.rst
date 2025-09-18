I/O Operations
==============

This module provides convenient utilities for file input/output operations, model checkpoint management, and data serialization in PyTorch projects.

Overview
--------

The I/O module includes three main components:

* **Model Management**: Save and load PyTorch model checkpoints with automatic versioning
* **File System**: Scan directories and check file existence with flexible options  
* **Data Serialization**: Handle JSON and CSV files with simple load/save functions

Quick Start
-----------

.. code-block:: python

    from torchutils.io import ModelSaver, load_json, scan_files
    
    # Save model checkpoints automatically
    saver = ModelSaver("./checkpoints", save_best=True)
    saver.save(model, accuracy_score, epoch)
    
    # Load configuration
    config = load_json("config.json")
    
    # Find all Python files
    py_files = scan_files("./src", suffix=".py", recursive=True)

Model Checkpoint Management
---------------------------

The :class:`ModelSaver` class provides intelligent checkpoint management for training workflows.

**Key Features:**

* Automatic file naming with scores and epochs
* Keep only the N best checkpoints to save disk space
* Save latest and best model copies for easy access
* Atomic saves to prevent corrupted files
* Distributed training support

**Example Usage:**

.. code-block:: python

    # Create model saver
    saver = ModelSaver(
        dirname="./checkpoints",
        filename_prefix="resnet",
        score_name="accuracy", 
        n_saved=3,
        save_best=True,
        mode="max"  # higher scores are better
    )
    
    # During training loop
    for epoch in range(100):
        # ... train model ...
        accuracy = evaluate(model)
        saver.save(model, accuracy, epoch)
    
    # Load best checkpoint
    best_path = saver.best_checkpoint
    model.load_state_dict(torch.load(best_path))

.. autoclass:: torchutils.io.model_saver.ModelSaver
   :members:
   :show-inheritance:

File System Utilities
---------------------

Simple functions to work with files and directories.

**Functions:**

* :func:`check_exists` - Check if files or folders exist
* :func:`scan_files` - Find files with optional filtering
* :func:`scan_folders` - Find directories with optional filtering

**Example Usage:**

.. code-block:: python

    from torchutils.io import check_exists, scan_files, scan_folders
    
    # Check if required files exist
    if check_exists(["data.csv", "config.json"], mode="all"):
        print("All files found!")
    
    # Find all image files recursively
    images = scan_files("./dataset", suffix=(".jpg", ".png"), recursive=True)
    
    # Get experiment directories
    exp_dirs = scan_folders("./experiments", recursive=False)

.. automodule:: torchutils.io.filesystem
   :members:
   :show-inheritance:

Data Serialization
------------------

Load and save JSON and CSV files with minimal code.

**Functions:**

* :func:`load_json` / :func:`save_json` - Handle JSON files
* :func:`load_csv` / :func:`save_csv` - Handle CSV files with optional headers and type conversion

**Example Usage:**

.. code-block:: python

    from torchutils.io import load_json, save_json, load_csv, save_csv
    
    # JSON operations
    config = load_json("config.json")
    save_json("results.json", {"accuracy": 0.95}, overwrite=True)
    
    # CSV operations  
    data = load_csv("scores.csv", skip_rows=1, converter=float)
    save_csv("output.csv", [[1, 0.9], [2, 0.8]], header=["id", "score"])

.. automodule:: torchutils.io.serialize
   :members:
   :show-inheritance: