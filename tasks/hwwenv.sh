#!/usr/bin/env bash

action() {
    local base="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && /bin/pwd )"
    local parent="$( dirname "$base" )"

    # software variables
    export LAW_HOME="$base/.law"
    export LAW_CONFIG_FILE="$base/law.cfg"
    export LUIGI_CONFIG_PATH="$base/luigi.cfg"

    # analysis variables
    export ANALYSIS_BASE="$base"
    export ANALYSIS_PARENT="$parent"
    export ANALYSIS_LOCAL_STORE="$ANALYSIS_BASE/data"
    export ANALYSIS_BASE_CONFIG="$ANALYSIS_BASE/analysis/MSSM_HWW.yaml"

    # storage path variables (EOS,...)
    # target path
    export ANALYSIS_DATA_PATH_TARGET="/eos/user/m/mfackeld/summer16/"
    # source path
    export ANALYSIS_DATA_PATH_SOURCE="/eos/cms/store/group/phys_higgs/cmshww/amassiro/Full2016_Apr17/"
    # tmp dir
    export ANALYSIS_TMP_DIR="/user/fackeldey/ml_tmp/"


}
action
