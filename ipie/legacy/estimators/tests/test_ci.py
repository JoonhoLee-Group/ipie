import pytest

from ipie.legacy.estimators.ci import simple_fci, simple_fci_bose_fermi
from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.systems.hubbard_holstein import HubbardHolstein
from ipie.legacy.systems.ueg import UEG


@pytest.mark.unit
def test_ueg():
    sys = UEG({"rs": 2, "nup": 2, "ndown": 2, "ecut": 0.5})
    ham = HamUEG(sys, {"rs": 2, "nup": 2, "ndown": 2, "ecut": 0.5})
    ham.ecore = 0
    eig, evec = simple_fci(sys, ham)
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
        "lang_firsov": False,
        "xpbc": True,
        "ypbc": True,
    }
    system = HubbardHolstein(options, verbose=True)
    ham = HubbardHolstein(options, verbose=True)
    (eig, evec), H = simple_fci_bose_fermi(system, ham, nboson_max=20, hamil=True)
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
        "lang_firsov": False,
        "xpbc": True,
        "ypbc": True,
    }
    system = HubbardHolstein(options, verbose=True)
    ham = HubbardHolstein(options, verbose=True)
    (eig, evec), H = simple_fci_bose_fermi(system, ham, nboson_max=20, hamil=True)
    assert eig[0] == pytest.approx(-4.642361166625703)


if __name__ == "__main__":
    test_hubbard_holstein()
