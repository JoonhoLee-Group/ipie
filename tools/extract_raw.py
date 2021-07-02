#!/usr/bin/env python

import h5py
import numpy
import sys
from pauxy.analysis.extraction import extract_mixed_estimates

data = extract_mixed_estimates(sys.argv[1])
print(data.to_string(index=False))
