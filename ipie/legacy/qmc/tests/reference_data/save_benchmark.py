import sys
import json

from ipie.analysis.extraction import extract_test_data_hdf5

if __name__ == "__main__":
    data = extract_test_data_hdf5("estimates.0.h5")
    json.dump(data, open("benchmark.json", "w"))
