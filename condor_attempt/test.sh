#!/bin/bash
# file name: test.sh

cat /etc/centos-release
source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos7-gcc11-opt/setup.sh

echo "Starting environment"
cd /eos/user/y/ymaidann/eth_project/Spanet_project
source SPANet/spanet-lcg103cuda-7/bin/activate
cd SPANet
pwd

echo "Training"
echo "python -m spanet.train -of options_files/$1 --epochs $2 --gpus $3"
python -m spanet.train -of options_files/$1 --epochs $2 --gpus $3
