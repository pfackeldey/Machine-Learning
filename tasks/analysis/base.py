# -*- coding: utf-8 -*-

import os

import luigi
import law

import law.contrib.htcondor

class DataSets():
    def __init__():
        self.datarun = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.selection = "__wwSel"
        self.shifts = ["JESdo", "JESup", "LepElepTdo", "LepMupTdo", "METdo", "METup", "PS", "PUdo", "PUup", "UEdo", "UEup", ""]
        self.base_path_mc = "Apr2017_summer16/lepSel__MCWeights__bSFLpTEffMulti__cleanTauMC__l2loose__hadd__l2tightOR__LepTrgFix__dorochester__formulasMC__"
        self.base_path_data = ["Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__hadd__l2tightOR__formulasDATA"]
        self.base_path_wjets = ["Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__multiFakeW__formulasFAKE__hadd"]


        def create_data_directories(self):
            return [self.base_path_data[0] + run[0] + self.base_path_data[1] + self.selection for run in self.datarun]

        def create_wjets_directories(self):
            return [self.base_path_data[0] + run[0] + self.base_path_data[1] + self.selection for run in self.datarun]


        def create_mc_directories(self):
            return [self.base_path_mc + shift + self.selection for shit in self.shifts]

        def directories(self):
            return self.create_data_directories + self.create_wjets_directories + self.create_mc_directories


class Task(Datasets, law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """

    version = luigi.Parameter()

    def store_parts(self):
        return (self.__class__.__name__, self.version)

    def local_path(self, *path):
        # ANALYSIS_DATA_PATH is defined in setup.sh
        """
        TODO: ENV has to be updated!!!
        """
        parts = (os.getenv("ANALYSIS_DATA_PATH_TARGET"),) + self.store_parts() + path
        return os.path.join(*parts)

    def local_target(self, *path):
        return law.LocalFileTarget(self.local_path(*path))

class HTCondorWorkflow(law.contrib.htcondor.HTCondorWorkflow):
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the VISPA environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """

    htcondor_gpus = luigi.IntParameter(default=law.NO_INT, significant=False, description="number "
        "of GPUs to request on the VISPA cluster, leave empty to use only CPUs")

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    def htcondor_create_job_file_factory(self):
        # tell the factory, which is responsible for creating our job files,
        # that the files are not temporary, i.e., it should not delete them after submission
        factory = super(HTCondorWorkflow, self).htcondor_create_job_file_factory()
        factory.is_tmp = False
        return factory

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path(__file__, "bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        # render_data is rendered into all files sent with a job
        config.render_variables["analysis_path"] = os.getenv("ANALYSIS_PATH")
        # copy the entire environment
        config.custom_content.append(("getenv", "true"))
        # tell the job config if GPUs are requested
        if not law.is_no_param(self.htcondor_gpus):
            config.custom_content.append(("request_gpus", self.htcondor_gpus))
        return config
