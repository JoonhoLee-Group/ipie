from ipie.legacy.hamiltonians.hubbard import get_strip


def correlation_function(filename, name, iy):
    data = h5py.File(filename, "r")
    md = json.loads(data["metadata"][:][0])
    nx = int(md["system"]["nx"])
    ny = int(md["system"]["ny"])
    output = data[name + "/correlation"][:]
    columns = ["hole", "hole_err", "spin", "spin_err"]
    h, herr = get_strip(output[0], output[1], iy[0], nx, ny)
    s, serr = get_strip(output[2], output[3], iy[0], nx, ny, True)
    results = pd.DataFrame(
        {"hole": h, "hole_err": herr, "spin": s, "spin_err": serr}, columns=columns
    )

    return results


def analysed_energies(filename, name):
    data = h5py.File(filename, "r")
    md = json.loads(data["metadata"][:][0])
    dt = md["qmc"]["dt"]
    output = data[name + "/estimates"][:]
    columns = data[name + "/headers"][:]
    results = pd.DataFrame(output, columns=columns)

    return results
