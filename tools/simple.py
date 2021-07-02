#!/usr/bin/env python

from pauxy.analysis import blocking
import glob
import sys
import pandas as pd

start_time = float(sys.argv[1])
files = glob.glob(sys.argv[2:][0])

data = blocking.analyse_simple(files, start_time)
pd.options.display.float_format = '{:,.8e}'.format
print (data)
# print (data[['E', 'EKin', 'EKin_error', 'EPot', 'EPot_error', 'E_T', 'E_error', 'dt', 'ecut', 'free_projection', 'ndown', 'nup', 'nwalkers', 'rs']].sort_values(['rs', 'ecut']).to_string())
