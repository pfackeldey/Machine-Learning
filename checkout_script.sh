#!/usr/bin/env bash
# cd home
cd ~

python36 -m pip install virtualenv

python36 -m virtualenv sw_base --distribute

cd sw_base

# clone repository
git clone https://github.com/CMSAachen3B/Machine-Learning.git

# activate virtualenv
source bin/activate

cd Machine-Learning

# install more python packages
pip install -r requirements.txt

