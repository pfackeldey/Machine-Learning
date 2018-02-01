import luigi
import os
import sys
import subprocess
import yaml

from preprocessing.createTrainingsset import createTrainingsset
from training.rootToNumpy import rootToNumpy

class CreateTrainingsset(luigi.Task):
    config_path = luigi.Parameter()
    num_fold = 1

    def __init__(self, *args, **kwargs):
        super(CreateTrainingsset, self).__init__(*args, **kwargs)

    def requires(self):
        return None

    def output(self):
        config = yaml.load(open(self.config_path, "r"))
        output_file = os.path.join(config["output_path_creation"], "fold{}_{}".format(self.num_fold, config["output_filename"]))
        return luigi.LocalTarget(output_file)

    def run(self):
        createTrainingsset(self.config_path)

class RootToNumpy(luigi.Task):
    config_path = luigi.Parameter()
    num_fold = 1

    def __init__(self, *args, **kwargs):
        super(RootToNumpy, self).__init__(*args, **kwargs)

    def requires(self):
        return CreateTrainingsset(self.config_path)

    def output(self):
        return luigi.LocalTarget("./arrays/weights_fold{}.npy".format(self.num_fold))

    def run(self):
        rootToNumpy(self.config_path)

if __name__ == '__main__':
    luigi.run()
