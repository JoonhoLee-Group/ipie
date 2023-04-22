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
err_3=$?
cd 08-custom_observable
python run_afqmc.py
err_8=$?
total_error=$(($err_1 + $err_2 + $err_3 + $err_8))
echo "Finished running example 3."

if [[ $total_error > 0 ]]; then
    echo "Error running examples."
    exit 1
else
    exit 0
fi
