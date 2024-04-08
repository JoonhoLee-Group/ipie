from ipie.utils.mpi import make_splits_displacements
import h5py
import numpy as np


def split_cholesky(ham_filename, nmembers):
    #
    with h5py.File(ham_filename, "r") as source_file:
        dataset = source_file["LXmn"][()]  # pylint: disable=no-member
        num_chol = dataset.shape[0]
    split_sizes, displacements = make_splits_displacements(num_chol, nmembers)
    assert isinstance(dataset, np.ndarray), "Dataset should be a numpy array"
    dataset = dataset.transpose(1, 2, 0).reshape(-1, num_chol)  # pylint: disable=no-member

    for i, (size, displacement) in enumerate(zip(split_sizes, displacements)):
        print(i, size, displacement)
        row_start = displacement
        row_end = displacement + size

        with h5py.File(f"chol_{i}.h5", "w") as target_file:
            target_file.create_dataset("chol", data=dataset[:, row_start:row_end])

    # Done
    print("Splitting complete.")
