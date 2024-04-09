from ipie.utils.mpi import make_splits_displacements
import h5py
import numpy as np


def split_cholesky(ham_filename: str, nmembers: int, verbose=True):
    """
    This function calculates the splits and displacements needed to distribute the
    Cholesky vectors among the members and  splits the Cholesky decomposed Hamiltonian
    vectors stored in an HDF5 file among a given number of members
    (e.g., GPU cards to distribute total cholesky)

    Parameters
    ----------
    ham_filename : str
        The filename of the HDF5 file containing the total Cholesky (naux, nbas, nbas)
    nmembers : int
        The number of members among which the Cholesky vectors will be distributed.
    """
    with h5py.File(ham_filename, "r") as source_file:
        # for huge chol file, should read in slices at one time instead of this
        dataset = np.array(source_file["LXmn"][:])
        num_chol = dataset.shape[0]
    split_sizes, displacements = make_splits_displacements(num_chol, nmembers)
    dataset = dataset.transpose(1, 2, 0).reshape(-1, num_chol)

    for i, (size, displacement) in enumerate(zip(split_sizes, displacements)):
        # Prepare row indices for slicing
        row_start = displacement
        row_end = displacement + size
        with h5py.File(f"chol_{i}.h5", "w") as target_file:
            target_file.create_dataset("chol", data=dataset[:, row_start:row_end])
        if verbose:
            print(f"# Split {i}: Size {size}, Displacement {displacement}")

    if verbose:
        print("# Splitting complete.")
