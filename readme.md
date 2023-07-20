Structure:
- data/: contains all data related files(datasets,utils for data,...)
- logging/: contains code for parsing, logging and experiment tracking
- models/: contains model implementations and maybe common modules
- notebooks/: contains various notebooks for experimentations/visualizations
- tests/ contains tests for various modules
- config.py: global data and data paths
- main.py: entry point 
- run_tests.py: entry point for tests

Practices:
- Inside modules/packages the imports are done using full paths(from Data.utils import ..) even if utils is a twin module
- packages/modules are lower snake case while classes are Capital camel case
- Models should provide a train_step, val_step, test_step & get optimizer
- Models receive hyperparameters and training details from a config.yml file
- Before each experiment the current state of models & code should be saved by a commit, the commit hash will be included in the logging folder
