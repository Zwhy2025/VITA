#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


sudo bash $SCRIPT_DIR/install_base.sh
sudo bash $SCRIPT_DIR/install_dev.sh
sudo bash $SCRIPT_DIR/install_runtime.sh
sudo bash $SCRIPT_DIR/install_graphics.sh

bash $SCRIPT_DIR/install_conda.sh 