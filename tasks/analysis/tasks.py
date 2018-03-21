# -*- coding: utf-8 -*-
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

from analysis.base import Task, HTCondorWorkflow
from analysis.datasets2016 import DataSets2016

class FetchData(DataSets2016, Task, law.LocalWorkflow):

    src = os.getenv("ANALYSIS_DATA_PATH_SOURCE")

    def create_branch_map(self):
        return {i: dir for i, dir in zip(range(len(self.directories())),self.directories())}

    def output(self):
        return self.local_target(self.branch_data)

    @law.decorator.log
    def run(self):
        # copy all dirs defined in DataSets2016
        import subprocess
        # create all missing directories
        subprocess.call(["mkdir", "-p", self.local_path(self.branch_data)])
        # copy files to created cirectories
        for item in os.listdir(self.src + self.branch_data):
            subprocess.call(["cp", "-r", os.path.join(self.src + self.branch_data, item), self.local_path(self.branch_data)])


class CreateTrainingsset(HTCondorWorkflow, law.SandboxTask):
    config_path = os.getenv("ANALYSIS_BASE_CONFIG")
    num_fold = 1

    sandbox = "docker::pfackeldey/hww"
    force_sandbox = True

    def __init__(self, *args, **kwargs):
        super(CreateTrainingsset, self).__init__(*args, **kwargs)

    def requires(self):
        return FetchData()

    def output(self):
        """
        Find a more convenient solution?
        """
        config = yaml.load(open(self.config_path, "r"))
        output_file = os.path.join(config["output_path_creation"], "fold{}_{}".format(self.num_fold, config["output_filename"]))
        return local_target(output_file)

    def run(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.getenv("ANALYSIS_BASE")))
        from preprocessing.createTrainingsset import createTrainingsset
        createTrainingsset(self.config_path)

class ConvertData(HTCondorWorkflow, law.SandboxTask):
    config_path = os.getenv("ANALYSIS_BASE_CONFIG")
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
        import sys
        sys.path.insert(0, os.path.dirname(os.getenv("ANALYSIS_BASE")))
        from preprocessing.rootToNumpy import rootToNumpy
        rootToNumpy(self.config_path)
