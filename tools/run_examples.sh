#!/bin/bash
tool_dir=$(pwd)/tools
if [[ ! -d examples ]]; then
    echo "Error: Run from pie base directory."
    exit 1
fi
cd examples
root_dir=$(pwd)
cd 01-simple
python scf.py
python $tool_dir/pyscf/pyscf_to_ipie.py -i scf.chk
echo "Finished running example 1."
err_1=$?
cd $root_dir
cd 02-multi_determinant
python scf.py
python $tool_dir/pyscf/pyscf_to_ipie.py -i scf.chk --mcscf
err_2=$?
cd $root_dir
echo "Finished running example 2."
cd 03-custom_observable
python run_afqmc.py
cd $root_dir
err_3=$?
echo "Finished running example 3."
cd 04-s2_observable
python run_afqmc.py
cd $root_dir
err_4=$?
echo "Finished running example 4."
cd 05-frozen_core
python scf.py
python $tool_dir/pyscf/pyscf_to_ipie.py -i scf.chk -j input.json --frozen-core 5
cd $root_dir
err_5=$?
echo "Finished running example 5."
cd 07-custom_trial
python run_afqmc.py
err_7=$?
cd $root_dir
echo "Finished running example 7."
cd 08-custom_walker
python run_afqmc.py
err_8=$?
echo "Finished running example 8."
cd $root_dir
cd 10-pyscf_interface
python run_afqmc.py
err_10=$?
echo "Finished running example 10."
cd $root_dir

total_error=$(($err_1 + $err_2 + $err_3 + $err_4 + $err_5 + $err_7 + $err_8 + $err_10))

if [[ $total_error > 0 ]]; then
    echo "Error running examples."
    exit 1
else
    echo "Examples finished successfully."
    exit 0
fi