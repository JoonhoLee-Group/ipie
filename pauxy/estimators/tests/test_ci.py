import pytest
from pauxy.systems.ueg import UEG
from pauxy.systems.hubbard_holstein import HubbardHolstein
from pauxy.estimators.ci import simple_fci, simple_fci_bose_fermi


@pytest.mark.unit
def test_ueg():
    sys = UEG({'rs': 2, 'nup': 2, 'ndown': 2, 'ecut': 0.5})
    sys.ecore = 0
    eig, evec = simple_fci(sys)
    assert len(eig) == 441
    assert eig[0] == pytest.approx(1.327088181107)
    assert eig[231] == pytest.approx(2.883365264420)
    assert eig[424] == pytest.approx(3.039496944900)
    assert eig[-1] == pytest.approx(3.207573492596)

@pytest.mark.unit
def test_hubbard_holstein():
    options = {
	    "name": "HubbardHolstein",
	    "nup": 1,
	    "ndown": 1,
	    "nx": 2,
	    "ny": 1,
	    "U": 0.0,
	    "w0": 0.8,
	    "lambda": 0.5,
	    "lang_firsov":False,
	    "xpbc" :True,
	    "ypbc" :True
    }
    system = HubbardHolstein (options, verbose=True)
    (eig, evec), H = simple_fci_bose_fermi(system, nboson_max=20, hamil=True)
    assert eig[0] == pytest.approx(-6.232530237466693)
    
    options = {
	    "name": "HubbardHolstein",
	    "nup": 1,
	    "ndown": 1,
	    "nx": 3,
	    "ny": 1,
	    "U": 4.0,
	    "w0": 0.8,
	    "lambda": 0.5,
	    "lang_firsov":False,
	    "xpbc" :True,
	    "ypbc" :True
    }
    system = HubbardHolstein (options, verbose=True)
    (eig, evec), H = simple_fci_bose_fermi(system, nboson_max=20, hamil=True)
    assert eig[0] == pytest.approx(-4.642361166625703)


if __name__=="__main__":
    test_hubbard_holstein()
