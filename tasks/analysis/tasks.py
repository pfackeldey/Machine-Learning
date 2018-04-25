# -*- coding: utf-8 -*-

import re
from collections import OrderedDict
from array import array

import luigi
import six
import law
law.contrib.load("numpy", "root", "wlcg", "tasks")

from analysis.base import Task, ConfigTask, ProcessTask, HTCondorWorkflow


class CopyFiles(ProcessTask):

    selection = "__wwSel"

    shift = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(CopyFiles, self).__init__(*args, **kwargs)

        if self.is_data:
            self.base_path = "Apr2017_Run2016{}_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__hadd__l2tightOR__formulasDATA"
        elif self.is_wjets:
            self.base_path = "Apr2017_Run2016{}_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__multiFakeW__formulasFAKE__hadd"
        else:
            self.base_path = "Apr2017_summer16/lepSel__MCWeights__bSFLpTEffMulti__cleanTauMC__l2loose__hadd__l2tightOR__LepTrgFix__dorochester__formulasMC"
            if self.shift:
                self.base_path += "__{}".format(self.shift)

    def store_parts(self):
        # [:-1] skips the process name which is added by the ProcessTask
        return super(CopyFiles, self).store_parts()[:-1] + ("orig", self.base_path + self.selection)

    def output(self):
        def target(path):
            base = path.rsplit("/")[-1]
            t = self.wlcg_target(base)
            if not self.is_mc:
                # extract the run period and re-format it into the target path (s.a.)
                run = re.search("Run2016(\w)", base).group(1)
                t.path = t.path.format(run)
            return t

        return law.SiblingFileCollection([
            target(path)
            for path in self.process_config["files"]
        ])

    def run(self):
        coll = self.output()
        coll.dir.touch()

        progress_cb = self.create_progress_callback(len(coll))

        for i, dst in enumerate(coll.targets):
            if dst.exists():
                continue

            src = law.WLCGFileTarget(
                "/".join(dst.path.rsplit("/", 3)[-3:]), fs="eos_phys_higgs_fs")
            with self.publish_step("upload file {} ...".format(i)):
                with dst.localize("w", cache=False) as dst_tmp:
                    src.copy_to_local(dst_tmp, cache=False)

            progress_cb(i)


class CopyFilesWrapper(ConfigTask, law.WrapperTask):
    """
shifts = [
    "JESdo", "JESup", "LepElepTdo", "LepMupTdo", "METdo", "METup", "PS", "PUdo", "PUup", "UEdo",
    "UEup",
]

def requires(self):
    reqs = [CopyFiles.req(self, process=process) for process in self.config["processes"]]
    for task in list(reqs):
        if task.is_mc:
            reqs += [CopyFiles.req(task, shift=shift) for shift in self.shifts]
    return reqs
"""

    def requires(self):
        return [CopyFiles.req(self, process=process) for process in self.config["processes"]]


class CreateTrainingset(ProcessTask, HTCondorWorkflow, law.LocalWorkflow):

    def create_branch_map(self):
        return self.process_config["files"]

    def workflow_requires(self):
        reqs = super(CreateTrainingset, self).workflow_requires()
        reqs["data"] = CopyFiles.req(self)
        return reqs

    def requires(self):
        return CopyFiles.req(self)

    def output(self):
        return [self.wlcg_target("data_fold{}_{}.root".format(i, self.branch)) for i in range(2)]

    def run(self):
        import ROOT
        ROOT.PyConfig.IgnoreCommandLineOptions = True
        ROOT.gROOT.SetBatch()

        inp = self.input().targets[self.branch]
        outputs = self.output()
        outputs[0].parent.touch()

        with inp.load("READ", formatter="root") as tfile_in:
            tree_in = tfile_in.Get("latino")

            for i, outp in enumerate(outputs):
                cut_string = "({CUT_STRING}*({EVENT_BRANCH}%2=={NUM_FOLD}))".format(
                    EVENT_BRANCH=self.config["event_branch"],
                    NUM_FOLD=i,
                    CUT_STRING=self.process_config["cut_string"]
                )
                with outp.localize("w") as outp_tmp:
                    with outp_tmp.load("RECREATE", formatter="root") as tfile_out:
                        tfile_out.cd()

                        # skimming
                        tree_out = tree_in.CopyTree(cut_string)
                        tree_out.SetName(self.process_config["class"])

                        # append a new branch
                        formula = ROOT.TTreeFormula("training_weight",
                                                    self.process_config["weight_string"], tree_out)
                        training_weight = array("f", [-999.0])
                        branch_training_weight = tree_out.Branch(
                            self.config["training_weight_branch"], training_weight,
                            self.config["training_weight_branch"] + "/F")
                        for i_event in range(tree_out.GetEntries()):
                            tree_out.GetEntry(i_event)
                            training_weight[0] = formula.EvalInstance()
                            branch_training_weight.Fill()

                        tfile_out.cd()
                        tree_out.Write()


class MergeTrainingset(ConfigTask):

    def requires(self):
        return {
            process: CreateTrainingset.req(self, process=process)
            for process in self.config["processes"]
        }

    def output(self):
        return [self.wlcg_target("data_fold{}.root".format(i)) for i in range(2)]

    def run(self):
        inputs = self.input()
        outputs = self.output()
        outputs[0].parent.touch()

        for i, outp in enumerate(outputs):
            tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
            tmp_dir.touch()

            names = []
            for inps in six.itervalues(inputs):
                for _inps in six.itervalues(inps["collection"].targets):
                    name = _inps[i].unique_basename
                    names.append(name)
                    _inps[i].copy_to_local(tmp_dir.child(name, type="f"))

            with outp.localize("w") as tmp:
                cmd = ["hadd", "-f0", tmp.path] + names
                with self.publish_step("hadding fold {} ...".format(i)):
                    code = law.util.interruptable_popen(
                        cmd, cwd=tmp_dir.path)[0]
                    if code != 0:
                        raise Exception("hadd failed")


class NumpyConversion(ConfigTask):

    def requires(self):
        return MergeTrainingset.req(self)

    def output(self):
        return [self.local_target("data_fold{}.npz".format(i)) for i in range(2)]

    def run(self):
        import numpy as np
        import root_numpy

        inputs = self.input()
        outputs = self.output()
        outputs[0].parent.touch()

        features = self.config["features"]
        classes = self.config["classes"]

        for i, (inp, outp) in enumerate(zip(inputs, outputs)):
            x, y, w = [], [], []

            with inp.load(formatter="root") as tfile:
                for i_class, class_ in enumerate(classes):
                    tree = tfile.Get(class_)
                    if not tree:
                        raise Exception("Tree %s not found" % class_)

                    # Get inputs for this class
                    x_class = np.zeros((tree.GetEntries(), len(features)))
                    x_conv = root_numpy.tree2array(tree, branches=features)
                    for i_feature, feature in enumerate(features):
                        x_class[:, i_feature] = x_conv[feature]
                    x.append(x_class)

                    # Get weights
                    w_class = np.zeros((tree.GetEntries(), 1))
                    w_conv = root_numpy.tree2array(
                        tree, branches=[self.config["training_weight_branch"]])
                    w_class[:, 0] = w_conv[self.config["training_weight_branch"]] * \
                        self.config["class_weights"][class_]
                    w.append(w_class)

                    # Get targets for this class
                    y_class = np.zeros((tree.GetEntries(), len(classes)))
                    y_class[:, i_class] = np.ones((tree.GetEntries()))
                    y.append(y_class)

                # Stack inputs, targets and weights to a Keras-readable dataset
                x = np.vstack(x)  # inputs
                y = np.vstack(y)  # targets
                w = np.vstack(w)  # weights
                w = np.squeeze(w)  # needed to get weights into keras

                outp.dump(x=x, y=y, w=w, formatter="numpy")
