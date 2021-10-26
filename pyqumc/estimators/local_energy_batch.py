import numpy
from pyqumc.estimators.local_energy import local_energy_G, local_energy_generic_cholesky
from pyqumc.utils.linalg import minor_mask, minor_mask4

# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial, iw = None):

    if (walker_batch.name == "SingleDetWalkerBatch"):
        return local_energy_single_det_batch(system, hamiltonian, walker_batch, trial, iw = iw)
    elif (walker_batch.name == "MultiDetTrialWalkerBatch" and trial.ortho_expansion == False):
        return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = iw)
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.ortho_expansion == True:
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = iw)
        return local_energy_multi_det_trial_wicks_batch(system, hamiltonian, walker_batch, trial, iw = iw)


def local_energy_multi_det_trial_wicks_batch(system, ham, walker_batch, trial, iw = None):
    assert(iw == None)

    nwalkers = walker_batch.nwalkers
    nbasis = ham.nbasis
    nchol = ham.nchol
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis*nbasis))
    Gb = walker_batch.Gb.reshape((nwalkers, nbasis*nbasis))
    e1bs = Ga.dot(ham.H1[0].ravel()) + Gb.dot(ham.H1[1].ravel()) + ham.ecore

    e2bs = []
    for iwalker in range(nwalkers):
        phi = walker_batch.phi[iwalker].copy() # walker wfn
        ovlpa0 = walker_batch.det_ovlpas[iwalker,0]
        ovlpb0 = walker_batch.det_ovlpbs[iwalker,0]
        ovlp0 = ovlpa0 * ovlpb0
        ovlp = walker_batch.ot[iwalker]

        # useful variables    
        G0a = walker_batch.G0a[iwalker]
        G0b = walker_batch.G0b[iwalker]
        Q0a = walker_batch.Q0a[iwalker]
        Q0b = walker_batch.Q0b[iwalker]
        CIa = walker_batch.CIa[iwalker]
        CIb = walker_batch.CIb[iwalker]
        G0 = [G0a, G0b]

        # print("G0a = {}".format(G0a))
        # contribution 1 (disconnected)
        cont1 = local_energy_generic_cholesky(system, ham, G0)[2] 

        # contribution 2 (half-connected, two-leg, one-body-like)
        # First, Coulomb-like term
        P0 = G0[0] + G0[1]
        Xa = numpy.einsum("m,mx->x", G0[0].ravel(), ham.chol_vecs)
        Xb = numpy.einsum("m,mx->x", G0[1].ravel(), ham.chol_vecs)
        
        LXa = ham.chol_vecs.dot(Xa)
        LXb = ham.chol_vecs.dot(Xb)
        LXa = LXa.reshape((nbasis,nbasis))
        LXb = LXb.reshape((nbasis,nbasis))

        # useful intermediate
        QCIGa = Q0a.dot(CIa).dot(G0a)
        QCIGb = Q0b.dot(CIb).dot(G0b)

        cont2_Jaa = numpy.sum(QCIGa * LXa)
        cont2_Jbb = numpy.sum(QCIGb * LXb)
        cont2_Jab = numpy.sum(QCIGb * LXa) + numpy.sum(QCIGa * LXb)
        cont2_J = cont2_Jaa + cont2_Jbb + cont2_Jab
        cont2_J *= (ovlp0/ovlp)

        # Second, Exchange-like term
        cont2_Kaa = 0.0 + 0.0j
        cont2_Kbb = 0.0 + 0.0j
        for x in range(nchol): 
            Lmn = ham.chol_vecs[:,x].reshape((nbasis, nbasis))
            LGL = Lmn.dot(G0a.T).dot(Lmn)
            cont2_Kaa -= numpy.sum(LGL*QCIGa) 

            LGL = Lmn.dot(G0b.T).dot(Lmn)
            cont2_Kbb -= numpy.sum(LGL*QCIGb)

        cont2_Kaa *= (ovlp0/ovlp)
        cont2_Kbb *= (ovlp0/ovlp)

        cont2_K = cont2_Kaa + cont2_Kbb

        Laa = numpy.einsum("iq,pj,ijx->qpx",Q0a, G0a, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)
        Lbb = numpy.einsum("iq,pj,ijx->qpx",Q0b, G0b, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)
      
        cont3 = 0.0 + 0.0j

        for jdet in range(1, trial.ndets):

            nex_a = len(trial.cre_a[jdet])
            nex_b = len(trial.cre_b[jdet])

            det_a = numpy.zeros((nex_a,nex_a), dtype=numpy.complex128)    
            det_b = numpy.zeros((nex_b,nex_b), dtype=numpy.complex128)    

            for iex in range(nex_a):
                det_a[iex,iex] = G0a[trial.cre_a[jdet][iex],trial.anh_a[jdet][iex]]
                for jex in range(iex+1, nex_a):
                    det_a[iex, jex] = G0a[trial.cre_a[jdet][iex],trial.anh_a[jdet][jex]]
                    det_a[jex, iex] = G0a[trial.cre_a[jdet][jex],trial.anh_a[jdet][iex]]
            for iex in range(nex_b):
                det_b[iex,iex] = G0b[trial.cre_b[jdet][iex],trial.anh_b[jdet][iex]]
                for jex in range(iex+1, nex_b):
                    det_b[iex, jex] = G0b[trial.cre_b[jdet][iex],trial.anh_b[jdet][jex]]
                    det_b[jex, iex] = G0b[trial.cre_b[jdet][jex],trial.anh_b[jdet][iex]]

            ovlpa = numpy.linalg.det(det_a) * trial.phase_a[jdet]
            ovlpb = numpy.linalg.det(det_b) * trial.phase_b[jdet]

            for x in range(nchol):
                La = Laa[:,:,x]
                Lb = Lbb[:,:,x]

                if (nex_a > 0 and nex_b > 0): # 2-leg opposite spin block
                    cofactor_a = numpy.zeros((nex_a-1, nex_a-1), dtype=numpy.complex128)
                    cofactor_b = numpy.zeros((nex_b-1, nex_b-1), dtype=numpy.complex128)

                    for iex in range(nex_a):
                        for jex in range(nex_a):
                            p = trial.cre_a[jdet][iex]
                            q = trial.anh_a[jdet][jex]
                            cofactor_a[:,:] = minor_mask(det_a, iex, jex)
                            det_cofactor_a = (-1)**(iex+jex)* numpy.linalg.det(cofactor_a)
                            for kex in range(nex_b):
                                for lex in range(nex_b):
                                    r = trial.cre_b[jdet][kex]
                                    s = trial.anh_b[jdet][lex]
                                    cofactor_b[:,:] = minor_mask(det_b, kex, lex)
                                    det_cofactor_b = (-1)**(kex+lex)* numpy.linalg.det(cofactor_b)
                                    cont3 += trial.coeffs[jdet].conj() * trial.phase_a[jdet] * trial.phase_b[jdet] * La[q,p]*Lb[s,r] * det_cofactor_a * det_cofactor_b

                if (nex_a == 2): # 4-leg same spin block aaaa
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_a[jdet][1]
                    s = trial.anh_a[jdet][1]
     
                    cont3 += trial.coeffs[jdet].conj() * trial.phase_a[jdet] * (La[q,p]*La[s,r]-La[q,r]*La[s,p]) * ovlpb

                elif (nex_a > 2):
                    cofactor = numpy.zeros((nex_a-2, nex_a-2), dtype=numpy.complex128)
                    for iex in range(nex_a):
                        for jex in range(nex_a):
                            p = trial.cre_a[jdet][iex]
                            q = trial.anh_a[jdet][jex]
                            for kex in range(iex+1, nex_a):
                                for lex in range(jex+1, nex_a):
                                    r = trial.cre_a[jdet][kex]
                                    s = trial.anh_a[jdet][lex]
                                    cofactor[:,:] = minor_mask4(det_a, iex, jex, kex, lex)
                                    det_cofactor = (-1)**(kex+lex+iex+jex)* numpy.linalg.det(cofactor)
                                    cont3 += trial.coeffs[jdet].conj() * trial.phase_a[jdet] * (La[q,p]*La[s,r]-La[q,r]*La[s,p]) * det_cofactor * ovlpb


                if (nex_b == 2): # 4-leg same spin block bbbb
                    p = trial.cre_b[jdet][0]
                    q = trial.anh_b[jdet][0]
                    r = trial.cre_b[jdet][1]
                    s = trial.anh_b[jdet][1]
     
                    cont3 += trial.coeffs[jdet].conj() * trial.phase_b[jdet] * (Lb[q,p]*Lb[s,r]-Lb[q,r]*Lb[s,p]) * ovlpa

                elif (nex_b > 2):
                    cofactor = numpy.zeros((nex_b-2, nex_b-2), dtype=numpy.complex128)
                    for iex in range(nex_b):
                        for jex in range(nex_b):
                            p = trial.cre_b[jdet][iex]
                            q = trial.anh_b[jdet][jex]
                            for kex in range(iex+1,nex_b):
                                for lex in range(jex+1,nex_b):
                                    r = trial.cre_b[jdet][kex]
                                    s = trial.anh_b[jdet][lex]
                                    cofactor[:,:] = minor_mask4(det_b, iex, jex, kex, lex)
                                    det_cofactor = (-1)**(kex+lex+iex+jex)* numpy.linalg.det(cofactor)
                                    cont3 += trial.coeffs[jdet].conj() * trial.phase_b[jdet] * (Lb[q,p]*Lb[s,r]-Lb[q,r]*Lb[s,p]) * det_cofactor * ovlpa

        cont3 *= (ovlp0/ovlp)

        e2bs += [cont1 + cont2_J + cont2_K + cont3]

    e2bs = numpy.array(e2bs, dtype=numpy.complex128)

    etot = e1bs + e2bs

    energy = numpy.zeros((walker_batch.nwalkers,3),dtype=numpy.complex128)
    energy[:,0] = etot
    energy[:,1] = e1bs
    energy[:,2] = e2bs
    return energy


def local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = None):
    energy = []
    ndets = trial.ndets
    if (iw == None):
        nwalkers = walker_batch.nwalkers
        # ndets x nwalkers
        for iwalker, (w, Ga, Gb, Ghalfa, Ghalfb) in enumerate(zip(walker_batch.det_weights, 
                            walker_batch.Gia, walker_batch.Gib, 
                            walker_batch.Gihalfa, walker_batch.Gihalfb)):
            denom = 0.0 + 0.0j
            numer0 = 0.0 + 0.0j
            numer1 = 0.0 + 0.0j
            numer2 = 0.0 + 0.0j
            for idet in range(ndets):
                # construct "local" green's functions for each component of A
                G = [Ga[idet], Gb[idet]]
                Ghalf = [Ghalfa[idet], Ghalfb[idet]]
                # return (e1b+e2b+ham.ecore, e1b+ham.ecore, e2b)
                e = list(local_energy_G(system, hamiltonian, trial, G, Ghalf=None))
                numer0 += w[idet] * e[0]
                numer1 += w[idet] * e[1]
                numer2 += w[idet] * e[2]
                denom += w[idet]
            # return (e1b+e2b+ham.ecore, e1b+ham.ecore, e2b)
            energy += [list([numer0/denom, numer1/denom, numer2/denom])]

    else:
        denom = 0.0 + 0.0j
        numer0 = 0.0 + 0.0j
        numer1 = 0.0 + 0.0j
        numer2 = 0.0 + 0.0j
        # ndets x nwalkers
        w = walker_batch.det_weights[iw]
        Ga = walker_batch.Gia[iw]
        Gb = walker_batch.Gib[iw]
        Ghalfa = walker_batch.Gihalfa[iw]
        Ghalfb = walker_batch.Gihalfb[iw]
        for idet in range(ndets):
            # construct "local" green's functions for each component of A
            G = [Ga[idet], Gb[idet]]
            Ghalf = [Ghalfa[idet], Ghalfb[idet]]
            # return (e1b+e2b+ham.ecore, e1b+ham.ecore, e2b)
            e = list(local_energy_G(system, hamiltonian, trial, G, Ghalf=None))
            numer0 += w[idet] * e[0]
            numer1 += w[idet] * e[1]
            numer2 += w[idet] * e[2]
            denom += w[idet]
        energy += [list([numer0/denom, numer1/denom, numer2/denom])]

    energy = numpy.array(energy, dtype=numpy.complex128)
    return energy

def local_energy_single_det_batch(system, hamiltonian, walker_batch, trial, iw = None):
    energy = []
    if (iw == None):
        nwalkers = walker_batch.nwalkers
        for idx in range(nwalkers):
            G = [walker_batch.Ga[idx],walker_batch.Gb[idx]]
            Ghalf = [walker_batch.Ghalfa[idx],walker_batch.Ghalfb[idx]]
            energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]

        energy = numpy.array(energy, dtype=numpy.complex128)
        return energy
    else:
        G = [walker_batch.Ga[iw],walker_batch.Gb[iw]]
        Ghalf = [walker_batch.Ghalfa[iw],walker_batch.Ghalfb[iw]]
        energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]
        energy = numpy.array(energy, dtype=numpy.complex128)
        return energy
    
