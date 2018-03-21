# -*- coding: utf-8 -*-

import os

class DataSets2016(object):
    """
    Class container providing the paths to all mc and data diretories of the HWW group.
    Additionally it provides functions to create branch maps in the main tasks.
    """
    def __init__(self, *args, **kwargs):
        super(DataSets2016, self).__init__(*args, **kwargs)

        self.datarun = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.selection = "__wwSel"
        self.shifts = ["__JESdo", "__JESup", "__LepElepTdo", "__LepMupTdo", "__METdo", "__METup", "__PS", "__PUdo", "__PUup", "__UEdo", "__UEup", ""]
        self.base_path_mc = "Apr2017_summer16/lepSel__MCWeights__bSFLpTEffMulti__cleanTauMC__l2loose__hadd__l2tightOR__LepTrgFix__dorochester__formulasMC"
        self.base_path_data = ["Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__hadd__l2tightOR__formulasDATA"]
        self.base_path_wjets = ["Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__multiFakeW__formulasFAKE__hadd"]


    def create_data_directories(self):
        return [self.base_path_data[0] + run[0] + self.base_path_data[1] + self.selection for run in self.datarun]

    def create_wjets_directories(self):
        return [self.base_path_wjets[0] + run[0] + self.base_path_wjets[1] + self.selection for run in self.datarun]

    def create_mc_directories(self):
        return [self.base_path_mc + shift + self.selection for shift in self.shifts]

    def directories(self):
        return self.create_data_directories() + self.create_wjets_directories() + self.create_mc_directories()
