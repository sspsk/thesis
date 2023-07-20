Structure:
-Data/: contains all data related files(datasets,utils for data,...)
-Logging/: contains code for parsing, logging and experiment tracking
-Models/: contains model implementations and maybe common modules
-Notebooks/: contains various notebooks for experimentations/visualizations
-config.py: global data and data paths
-main.py: entry point 

Practices:
-Inside modules/packages the imports are done using full paths(from Data.utils import ..) even if utils is a twin module
-Models should provide a train_step, val_step, test_step & get optimizer
-Models receive hyperparameters and training details from a config.yml file
-Before each experiment the current state of models & code should be saved by a commit, the commit hash will be included in the logging folder
