# Use this file only for testing

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../tools")
from pathmanager import PathManager
from tools import list_files


if __name__ == "__main__":
    list_files(os.path.dirname(__file__) + "/../..")
    path_manager: PathManager = PathManager()