# Developer Guidelines

## Developer Tools

We provide [dev/run_tests.py](../../../dev/run_tests.py) as a helper function to run
the ci workflow. This is useful for checking things other than the unit
tests which often we forget about (formatting, linting, integration
tests etc.)

``` bash
python dev/run_tests.py --help
```

Will print the available options. The option [\--all]{.title-ref} will
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
    @pytest.mark.gpu def test_my_special_test():
```

-   Currently it is advised to place gpu specific unit tests in
    filenames with \_gpu.py in the name.
-   To run the tests use:

``` bash
export IPIE_USE_GPU=1; mpirun -np 1 pytest -m gpu -sv
```

-   Note if running CPU test afterwards it may be necessary to clear the
    environment variable!


## Releasing a package

1. Change the version from 'X.Y.Z.dev0' to 'X.Y.Z'
2. Merge develop into main. Do not squash the merge (enables sensible release notes.)
3. Push tag 'X.Y.Z'.
4. Check actions and ensure build and publish steps run.
5. Create new branch and bump version to 'X.(Y+1).Z.dev0'.
6. Send PR for this branch into develop.
