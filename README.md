# Installation

Most of the code in this repo needs a functioning installation of tierpsy.
To make sure that this repo does not break the installation of tierpsy used for production, let's just install it again in a separate environment.

    # create environment
    conda create -n tierpsy_nn
    conda activate tierpsy_nn
    # prepare folder for repos
    cd ~
    mkdir behavgenom_repos
    cd behavgenom_repos
    # download source code
    git clone https://github.com/Tierpsy/tierpsy-tracker.git
    git clone https://github.com/luigiferiani/tierpsy-nn.git
    # install tierpsy
    cd tierpsy-tracker
    conda install --file requirements.txt -c conda-forge
    pip install -e .
    cd ..
    cd tierpsy-nn


Now the scripts in tierpsy-nn will be able to call upon functions from tierpsy.
