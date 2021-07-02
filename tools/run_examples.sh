#!/bin/bash
tool_dir=$(pwd)/tools
if [[ ! -d examples/generic ]]; then
    echo "Error: Run from pauxy base directory."
    exit 1
fi
cd examples/generic
root_dir=$(pwd)
cd 01-simple
python scf.py
python $tool_dir/pyscf/pyscf_to_pauxy.py -i scf.chk -o afqmc.h5
err_status=$?
cd $root_dir
cd 02-multi_determinant
python gen_msd.py
total_error=$(($? + $err_status))

if [[ $total_error > 0 ]]; then
    echo "Error running examples."
    exit 1
else
    exit 0
fi
