#!/usr/bin/env bash

action() {
    local base="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && /bin/pwd )"

    if [ "$ON_LX3B" != "1" ] || [ -z "$ANALYSIS_SOFTWARE_PATH" ]; then
        echo "not in lx3b or environment not set up, abort"
        return "1"
    fi

    echo "install software at $ANALYSIS_SOFTWARE_PATH"

    _install_pip() {
        pip install --ignore-installed --prefix "$ANALYSIS_SOFTWARE_PATH" "$1"
    }

    _install_pip six
    _install_pip luigi
    _install_pip git+https://github.com/riga/law.git
}
action "$@"
