import numpy
from pie.estimators.local_energy import local_energy_G, local_energy_generic_cholesky
from pie.utils.linalg import minor_mask, minor_mask4
from pie.utils.misc import is_cupy

# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial, iw = None):

    if (walker_batch.name == "SingleDetWalkerBatch"):
        if (is_cupy(walker_batch.phia)):
            return local_energy_single_det_batch_einsum(system, hamiltonian, walker_batch, trial, iw = iw)
        else:
            return local_energy_single_det_batch(system, hamiltonian, walker_batch, trial, iw = iw)
    elif (walker_batch.name == "MultiDetTrialWalkerBatch" and trial.wicks == False):
        return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = iw)
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == True:
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
        Xa = ham.chol_vecs.T.dot(G0[0].ravel()) #numpy.einsum("m,xm->x", G0[0].ravel(), ham.chol_vecs)
        Xb = ham.chol_vecs.T.dot(G0[1].ravel()) #numpy.einsum("m,xm->x", G0[1].ravel(), ham.chol_vecs)
        
        LXa = numpy.einsum("mx,x->m", ham.chol_vecs, Xa, optimize=True)
        LXb = numpy.einsum("mx,x->m", ham.chol_vecs, Xb, optimize=True)
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

        Laa = numpy.einsum("iq,pj,ijx->xqp",Q0a, G0a, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)
        Lbb = numpy.einsum("iq,pj,ijx->xqp",Q0b, G0b, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)
      
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
                La = Laa[x,:,:]
                Lb = Lbb[x,:,:]

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
    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        array = cupy.array
    else:
        array = numpy.array

    energy = []
    if (iw == None):
        nwalkers = walker_batch.nwalkers
        for idx in range(nwalkers):
            G = [walker_batch.Ga[idx],walker_batch.Gb[idx]]
            Ghalf = [walker_batch.Ghalfa[idx],walker_batch.Ghalfb[idx]]
            energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]

        energy = array(energy, dtype=numpy.complex128)
        return energy
    else:
        G = [walker_batch.Ga[iw],walker_batch.Gb[iw]]
        Ghalf = [walker_batch.Ghalfa[iw],walker_batch.Ghalfb[iw]]
        energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]
        energy = array(energy, dtype=numpy.complex128)
        return energy
    
def local_energy_single_det_batch_einsum(system, hamiltonian, walker_batch, trial, iw = None):

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        einsum = cupy.einsum
        zeros = cupy.zeros
        isrealobj = cupy.isrealobj
    else:
        einsum = numpy.einsum
        zeros = numpy.zeros
        isrealobj = numpy.isrealobj

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol
    
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis*nbasis))
    Gb = walker_batch.Gb.reshape((nwalkers, nbasis*nbasis))
    e1b = Ga.dot(hamiltonian.H1[0].ravel()) + Gb.dot(hamiltonian.H1[1].ravel()) + hamiltonian.ecore

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta*nbasis)

    if (isrealobj(trial._rchola)):
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.j * trial._rchola.dot(walker_batch.Ghalfa.imag.T) # naux x nwalkers
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.real.T) + 1.j * trial._rcholb.dot(walker_batch.Ghalfb.imag.T) # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)
        Xb = trial._rchola.dot(walker_batch.Ghalfb.T)

    ecoul = einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * einsum("xw,xw->w", Xa, Xb, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    GhalfaT_batch = walker_batch.Ghalfa.transpose(0,2,1).copy() # nw x nbasis x nocc
    GhalfbT_batch = walker_batch.Ghalfb.transpose(0,2,1).copy() # nw x nbasis x nocc

    Ta = zeros((nwalkers, nalpha,nalpha), dtype=numpy.complex128)
    Tb = zeros((nwalkers, nbeta,nbeta), dtype=numpy.complex128)

    exx  = zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(nchol):  # write a cython function that calls blas for this.
        rmi_a = trial._rchola[x].reshape((nalpha,nbasis))
        rmi_b = trial._rcholb[x].reshape((nbeta,nbasis))
        if (isrealobj(trial._rchola)):
            Ta[:,:,:].real = rmi_a.dot(GhalfaT_batch.real).transpose(1,0,2)
            Ta[:,:,:].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1,0,2)
            Tb[:,:,:].real = rmi_b.dot(GhalfbT_batch.real).transpose(1,0,2)
            Tb[:,:,:].imag = rmi_b.dot(GhalfbT_batch.imag).transpose(1,0,2)
        else:
            Ta[:,:,:] = rmi_a.dot(GhalfaT_batch).transpose(1,0,2)
            Tb[:,:,:] = rmi_b.dot(GhalfbT_batch).transpose(1,0,2)

        exx += einsum("wij,wji->w",Ta,Ta,optimize=True) + einsum("wij,wji->w",Tb,Tb,optimize=True)

    e2b = 0.5 * (ecoul - exx)

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

def local_energy_single_det_rhf_batch(system, hamiltonian, walker_batch, trial, iw = None):

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        einsum = cupy.einsum
        zeros = cupy.zeros
        isrealobj = cupy.isrealobj
    else:
        einsum = numpy.einsum
        zeros = numpy.zeros
        isrealobj = numpy.isrealobj

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol
    
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis*nbasis))
    e1b = 2.0 * Ga.dot(hamiltonian.H1[0].ravel()) + hamiltonian.ecore

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)

    if (isrealobj(trial._rchola)):
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.j * trial._rchola.dot(walker_batch.Ghalfa.imag.T) # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)

    ecoul = 2. * einsum("xw,xw->w", Xa, Xa, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    GhalfaT_batch = walker_batch.Ghalfa.transpose(0,2,1).copy() # nw x nbasis x nocc

    Ta = zeros((nwalkers, nalpha,nalpha), dtype=numpy.complex128)

    exx  = zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(nchol):  # write a cython function that calls blas for this.
        rmi_a = trial._rchola[x].reshape((nalpha,nbasis))
        if (isrealobj(trial._rchola)):
            Ta[:,:,:].real = rmi_a.dot(GhalfaT_batch.real).transpose(1,0,2)
            Ta[:,:,:].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1,0,2)
        else:
            Ta[:,:,:] = rmi_a.dot(GhalfaT_batch).transpose(1,0,2)

        exx += einsum("wij,wji->w",Ta,Ta,optimize=True)

    e2b = ecoul - exx

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy
