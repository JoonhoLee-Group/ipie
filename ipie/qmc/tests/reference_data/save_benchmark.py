import sys
import json

from ipie.analysis.extraction import extract_test_data_hdf5

if __name__ == '__main__':
    skip = 1 # save all the data
    data = extract_test_data_hdf5('estimates.0.h5', skip=skip)
    data['extract_skip_value'] = skip
    json.dump(data, open('reference.json', 'w'))
