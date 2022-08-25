import pytest

@pytest.fixture
def gpu_env(pytestconfig):
    markers_arg = pytestconfig.getoption('-m')
    print("markers arg: ", markers_arg, markers_arg == 'gpu')
    if markers_arg == 'gpu':
        from ipie.config import purge_ipie_modules, config
        config.update_option('use_gpu', True)
        purge_ipie_modules()
        config.update_option('use_gpu', True)
