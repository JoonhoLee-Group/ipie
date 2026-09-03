# Developer Guidelines

## Developer Tools

We provide [dev/run_tests.py](https://github.com/JoonhoLee-Group/ipie/blob/develop/dev/run_tests.py) as a helper script that runs
the same stages as the CI workflow (`.github/workflows/ci.yml`). This is
useful for checking things other than the unit tests which often we forget
about (formatting, linting, integration tests etc.)

All commands below are run from the repository root and require the
developer extra (`pip install -e ".[dev]"`, see {doc}`../installation`),
which provides `black`, `pylint`, `flynt` and `pytest-xdist`.

``` bash
python dev/run_tests.py --help
```

Will print the available options. The option `--all` will
run all the stages of the workflow which may take \~ 10 minutes or more.
A cheaper option (assuming no major code changes) is to run

``` bash
python dev/run_tests.py --black --pylint --pytest --flynt
```

Which should catch 99% of the CI errors.

## Code Formatting

-   Use black.

``` bash
python dev/run_tests.py --black
```

## Code Linting

-   Use pylint.

``` bash
python dev/run_tests.py --pylint
```

## Use fstrings

-   Use flynt.

``` bash
python dev/run_tests.py --flynt
```

## GPU

-   Due to current global config setup for device selection gpu unit
    tests need to be run separately from the bulk of the unit tests.
-   This is achieved by marking the unit test like so:

``` python
@pytest.mark.gpu
def test_my_special_test():
    ...
```

-   Currently it is advised to place gpu specific unit tests in
    filenames with \_gpu.py in the name.
-   To run the tests use:

``` bash
export IPIE_USE_GPU=1; mpirun -np 1 pytest -m gpu -sv
```

-   `IPIE_USE_GPU=1` is required: GPU-only tests such as the k-point ISDF
    estimator tests call `pytest.skip` when `config.get_option("use_gpu")`
    is not set, so running `pytest -m gpu` alone (as listed in
    {doc}`../installation`) silently skips them.
-   Note if running CPU test afterwards it may be necessary to clear the
    environment variable!


## Releasing a package

1. Change `__version__` in `ipie/_version.py` from 'X.Y.Z.dev0' to 'X.Y.Z'
   (`setup.py` reads the version from this file).
2. Merge develop into main. Do not squash the merge (enables sensible release notes.)
3. Push tag 'vX.Y.Z' (no dev0).
4. Check actions and ensure build and publish steps run.
5. Create new branch and bump version to 'X.(Y+1).Z.dev0'.
6. Send PR for this branch into develop.
