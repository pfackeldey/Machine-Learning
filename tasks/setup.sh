#!/usr/bin/env bash

action() {
    local base="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    echo "sourcing analysis setup from $base"

    export PYTHONPATH="$base:$PYTHONPATH"
    export LAW_HOME="$base/.law"
    export LAW_CONFIG_FILE="$base/law.cfg"
    export LUIGI_CONFIG_PATH="$base/luigi.cfg"

    export ANALYSIS_PATH="$base"
    export ANALYSIS_PARENT="$parent"
    export ANALYSIS_BASE_CONFIG="$ANALYSIS_PATH/analysis/MSSM_HWW.yaml"

    # detect environment
    if [[ "$( hostname )" == lx3b*.rwth-aachen.de ]]; then
        export ON_LX3B="1"
    fi
    if [[ "$( hostname )" == cluster.rz.RWTH-Aachen.DE ]]; then
        export ON_RZ="1"
    fi

    # environment dependend setup
    if [ "$ON_LX3B" = "1" ] || [ "$ON_CONDOR" = "1" ]; then
        export ANALYSIS_DATA_PATH="/net/scratch_cms3b/fackeldey/ml_data"
        export ANALYSIS_SOFTWARE_PATH="/net/scratch_cms3b/fackeldey/ml_software"
        export ANALYSIS_REMOTE_CACHE="$( [ "$ON_LX3B" = "1" ] && echo "/net/scratch_cms3b/fackeldey/ml_cache" )"

        # software from /net/scratch_cms
        for pkg in zlib-1.2.8 jsonc-test python-2.7.9 setuptools-34.1.1 pip-9.0.1 pycrypto-2.6.1 \
            pyparsing-2.1.10 packaging-16.8 appdirs-1.4.0 mock-1.3.0 root-5.34.34 numpy-1.13.0 \
            scipy-0.19.1 root_numpy-4.4.0 scikit_learn-0.19.1 h5py-2.7.1 hdf5-1.10.1 pyyaml-3.11 \
            gfal2-test; do
            source "/net/software_cms/vispa/sl6_local/$pkg/setup.sh"
        done

        # local software
        export PATH="$PATH:$ANALYSIS_SOFTWARE_PATH/bin"
        export PYTHONPATH="$PYTHONPATH:$ANALYSIS_SOFTWARE_PATH/lib/python2.7/site-packages"
    fi
    if [ "$ON_RZ" = "1" ]; then
        source /home/phys3b/Envs/keras_tf_sharedUsers/bin/activate
        #TODO
        # and other stuff ...
    fi

    source "$( law completion )"
}
action
