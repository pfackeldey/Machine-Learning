#!/usr/bin/env bash

action() {
    export ON_CONDOR="1"
    source "{{analysis_path}}/setup.sh"
}
action
