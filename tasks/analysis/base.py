# -*- coding: utf-8 -*-

import os

import luigi
import law
law.contrib.load("htcondor")


class Task(law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """

    def store_parts(self):
        return (self.__class__.__name__,)

    def local_path(self, *path):
        parts = (os.getenv("ANALYSIS_DATA_PATH"),) + self.store_parts() + path
        return os.path.join(*parts)

    def local_target(self, *path):
        return law.LocalFileTarget(self.local_path(*path))

    def wlcg_path(self, *path):
        parts = self.store_parts() + path
        return os.path.join(*parts)

    def wlcg_target(self, *path):
        return law.WLCGFileTarget(self.wlcg_path(*path))


class ConfigTask(Task):

    def __init__(self, *args, **kwargs):
        super(ConfigTask, self).__init__(*args, **kwargs)

        self.config = law.LocalFileTarget(
            law.util.rel_path(__file__, "MSSM_HWW.yaml")).load()


class ProcessTask(ConfigTask):

    process = luigi.Parameter(description="the process name")

    def __init__(self, *args, **kwargs):
        super(ProcessTask, self).__init__(*args, **kwargs)

        # validate process
        if self.process not in self.config["processes"]:
            raise ValueError("unknown process: {}".format(self.process))

        self.process_config = self.config["processes"][self.process]

        # some flags
        self.is_data = self.process.endswith("_data")
        self.is_wjets = self.process.endswith("_WJets")
        self.is_mc = not self.is_data and not self.is_wjets

    def store_parts(self):
        return super(ProcessTask, self).store_parts() + (self.process,)


class HTCondorWorkflow(law.HTCondorWorkflow):

    htcondor_logs = luigi.BoolParameter()

    outputs_siblings = True

    def __init__(self, *args, **kwargs):
        super(HTCondorWorkflow, self).__init__(*args, **kwargs)

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    def htcondor_create_job_file_factory(self):
        # tell the factory, which is responsible for creating our job files,
        # that the files are not temporary, i.e., it should not delete them after submission
        factory = super(HTCondorWorkflow,
                        self).htcondor_create_job_file_factory()
        factory.is_tmp = False
        return factory

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path(__file__, "..", "condor_bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        # render_data is rendered into all files sent with a job
        config.render_variables["analysis_path"] = os.getenv("ANALYSIS_PATH")

        # condor logs
        if self.htcondor_logs:
            config.stdout = "out.txt"
            config.stderr = "err.txt"
            config.log = "log.txt"

        return config

    def htcondor_output_postfix(self):
        self.get_branch_map()
        return "_{}To{}".format(self.start_branch, self.end_branch)

    def htcondor_use_local_scheduler(self):
        return True
