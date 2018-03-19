"""
Status: WIP!!!

Main missing parts: - submission of tasks to batch system
                    - more tasks... especially for training and evaluation and copying stuff (maybe even plotting)
                    - https://github.com/riga/law/blob/663448a34226f663c9a08a478aeee39227321172/law/sandbox/base.py for sandbox env variables
"""


import os
import law
import luigi

law.contrib.load("numpy", "root")

import sys
sys.path.insert(0, os.getenv("GLOBAL_ENV_ML_DIR"))

from preprocessing.createTrainingsset import createTrainingsset
from preprocessing.rootToNumpy import rootToNumpy

class FetchData(law.SandboxTask):

    def output(self):
        return self.local_target("data.root")

    @law.decorator.log
    def run(self):
        from shutil import copyfile
        """
        TODO: global env variables in setup.sh/hwwenv.sh with paths to all files of latino trees
              Set src, dst for MC, Data and all shifts...
        """
        copyfile(src, dst)

class CreateTrainingsset(law.SandboxTask):
    """
    define path to config in global env variable "GLOBAL_ENV" in setup.sh/hwwenv.sh
    """
    config_path = os.getenv("GLOBAL_ENV_CONFIG_PATH")
    num_fold = 1

    sandbox = "docker::pfackeldey/hww"
    force_sandbox = True

    def __init__(self, *args, **kwargs):
        super(CreateTrainingsset, self).__init__(*args, **kwargs)

    def requires(self):
        return None

    def output(self):
        """
        Find a more convenient solution?
        """
        config = yaml.load(open(self.config_path, "r"))
        output_file = os.path.join(config["output_path_creation"], "fold{}_{}".format(self.num_fold, config["output_filename"]))
        return local_target(output_file)

    def run(self):
        createTrainingsset(self.config_path)

class ConvertData(law.SandboxTask):
    """
    define path to config in global env variable "GLOBAL_ENV" in setup.sh/hwwenv.sh
    """
    config_path = os.getenv("GLOBAL_ENV_CONFIG_PATH")
    num_fold = 1

    sandbox = "docker::pfackeldey/hww"
    force_sandbox = True

    def __init__(self, *args, **kwargs):
        super(ConvertData, self).__init__(*args, **kwargs)

    def requires(self):
        return CreateTrainingsset(self.config_path)

    def output(self):
        """
        Find a more convenient solution...
        """
        return local_target("./arrays/weights_fold{}.npy".format(self.num_fold))

    def run(self):
        rootToNumpy(self.config_path)
