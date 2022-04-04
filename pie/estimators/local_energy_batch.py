import time
import numpy
from pie.estimators.local_energy import local_energy_G, local_energy_generic_cholesky
from pie.utils.linalg import minor_mask, minor_mask4
from pie.propagation.overlap import (
        get_overlap_one_det_wicks,
        compute_determinants_batched
        )
from pie.propagation.wicks_kernels import (
        get_det_matrix_batched,
        get_cofactor_matrix_batched,
        get_cofactor_matrix_4_batched,
        )
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
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == True and not trial.optimized:
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = iw)
        return local_energy_multi_det_trial_wicks_batch(system, hamiltonian, walker_batch, trial, iw = iw)
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == True and trial.optimized == True:
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = iw)
        return local_energy_multi_det_trial_wicks_batch_opt(system, hamiltonian, walker_batch, trial, iw = iw)

def local_energy_multi_det_trial_wicks_batch(system, ham, walker_batch, trial, iw=None):
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

        Laa = numpy.einsum("iq,pj,ijx->qpx",Q0a, G0a, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)
        Lbb = numpy.einsum("iq,pj,ijx->qpx",Q0b, G0b, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)

        cont3 = 0.0 + 0.0j

        for jdet in range(1, trial.ndets):

            nex_a = len(trial.cre_a[jdet])
            nex_b = len(trial.cre_b[jdet])

            ovlpa, ovlpb = get_overlap_one_det_wicks(nex_a, trial.cre_a[jdet], trial.anh_a[jdet], G0a,\
                nex_b, trial.cre_b[jdet], trial.anh_b[jdet], G0b)
            ovlpa *= trial.phase_a[jdet]
            ovlpb *= trial.phase_b[jdet]

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

            cphasea = trial.coeffs[jdet].conj() * trial.phase_a[jdet]
            cphaseb = trial.coeffs[jdet].conj() * trial.phase_b[jdet]
            cphaseab = trial.coeffs[jdet].conj() * trial.phase_a[jdet] * trial.phase_b[jdet]

            if (nex_a > 0 and nex_b > 0): # 2-leg opposite spin block
                if (nex_a == 1 and nex_b == 1):
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_b[jdet][0]
                    s = trial.anh_b[jdet][0]
                    cont3 += cphaseab * numpy.dot(Laa[q,p,:],Lbb[s,r,:])
                elif (nex_a == 2 and nex_b == 1):
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_a[jdet][1]
                    s = trial.anh_a[jdet][1]
                    t = trial.cre_b[jdet][0]
                    u = trial.anh_b[jdet][0]
                    cofactor = [cphaseab * G0a[r,s],
                    cphaseab * G0a[r,q],
                    cphaseab * G0a[p,s],
                    cphaseab * G0a[p,q]]
                    cont3 += numpy.dot(Laa[q,p,:],Lbb[u,t,:]) * cofactor[0]
                    cont3 -= numpy.dot(Laa[s,p,:],Lbb[u,t,:]) * cofactor[1]
                    cont3 -= numpy.dot(Laa[q,r,:],Lbb[u,t,:]) * cofactor[2]
                    cont3 += numpy.dot(Laa[s,r,:],Lbb[u,t,:]) * cofactor[3]
                elif (nex_a == 1 and nex_b == 2):
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_b[jdet][0]
                    s = trial.anh_b[jdet][0]
                    t = trial.cre_b[jdet][1]
                    u = trial.anh_b[jdet][1]
                    cofactor = [cphaseab * G0b[t,u],
                                cphaseab * G0b[t,s],
                                cphaseab * G0b[r,u],
                                cphaseab * G0b[r,s]]
                    cont3 +=  numpy.dot(Laa[q,p,:],Lbb[s,r,:]) * cofactor[0]
                    cont3 -=  numpy.dot(Laa[q,p,:],Lbb[u,r,:]) * cofactor[1]
                    cont3 -=  numpy.dot(Laa[q,p,:],Lbb[s,t,:]) * cofactor[2]
                    cont3 +=  numpy.dot(Laa[q,p,:],Lbb[u,t,:]) * cofactor[3]
                elif (nex_a == 2 and nex_b == 2):
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_a[jdet][1]
                    s = trial.anh_a[jdet][1]

                    t = trial.cre_b[jdet][0]
                    u = trial.anh_b[jdet][0]
                    v = trial.cre_b[jdet][1]
                    w = trial.anh_b[jdet][1]
                    cofactor = [cphaseab * G0a[r,s] * G0b[v,w],
                                cphaseab * G0a[r,q] * G0b[v,w],
                                cphaseab * G0a[p,s] * G0b[v,w],
                                cphaseab * G0a[p,q] * G0b[v,w],
                                cphaseab * G0a[r,s] * G0b[t,u],
                                cphaseab * G0a[r,q] * G0b[t,u],
                                cphaseab * G0a[p,s] * G0b[t,u],
                                cphaseab * G0a[p,q] * G0b[t,u],
                                cphaseab * G0a[r,s] * G0b[v,u],
                                cphaseab * G0a[r,q] * G0b[v,u],
                                cphaseab * G0a[p,s] * G0b[v,u],
                                cphaseab * G0a[p,q] * G0b[v,u],
                                cphaseab * G0a[r,s] * G0b[t,w],
                                cphaseab * G0a[r,q] * G0b[t,w],
                                cphaseab * G0a[p,s] * G0b[t,w],
                                cphaseab * G0a[p,q] * G0b[t,w]]
                    cont3 += numpy.dot(Laa[q,p,:],Lbb[u,t,:]) * cofactor[0]
                    cont3 -= numpy.dot(Laa[s,p,:],Lbb[u,t,:]) * cofactor[1]
                    cont3 -= numpy.dot(Laa[q,r,:],Lbb[u,t,:]) * cofactor[2]
                    cont3 += numpy.dot(Laa[s,r,:],Lbb[u,t,:]) * cofactor[3]

                    cont3 += numpy.dot(Laa[q,p,:],Lbb[w,v,:]) * cofactor[4]
                    cont3 -= numpy.dot(Laa[s,p,:],Lbb[w,v,:]) * cofactor[5]
                    cont3 -= numpy.dot(Laa[q,r,:],Lbb[w,v,:]) * cofactor[6]
                    cont3 += numpy.dot(Laa[s,r,:],Lbb[w,v,:]) * cofactor[7]

                    cont3 -= numpy.dot(Laa[q,p,:],Lbb[w,t,:]) * cofactor[8]
                    cont3 += numpy.dot(Laa[s,p,:],Lbb[w,t,:]) * cofactor[9]
                    cont3 += numpy.dot(Laa[q,r,:],Lbb[w,t,:]) * cofactor[10]
                    cont3 -= numpy.dot(Laa[s,r,:],Lbb[w,t,:]) * cofactor[11]

                    cont3 -= numpy.dot(Laa[q,p,:],Lbb[u,v,:]) * cofactor[12]
                    cont3 += numpy.dot(Laa[s,p,:],Lbb[u,v,:]) * cofactor[13]
                    cont3 += numpy.dot(Laa[q,r,:],Lbb[u,v,:]) * cofactor[14]
                    cont3 -= numpy.dot(Laa[s,r,:],Lbb[u,v,:]) * cofactor[15]

                elif (nex_a == 3 and nex_b == 1):
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_a[jdet][1]
                    s = trial.anh_a[jdet][1]
                    t = trial.cre_a[jdet][2]
                    u = trial.anh_a[jdet][2]
                    v = trial.cre_b[jdet][0]
                    w = trial.anh_b[jdet][0]
                    cofactor = [G0a[r,s]*G0a[t,u] - G0a[r,u]*G0a[t,s],
                                G0a[r,q]*G0a[t,u] - G0a[r,u]*G0a[t,q],
                                G0a[r,q]*G0a[t,s] - G0a[r,s]*G0a[t,q],
                                G0a[p,s]*G0a[t,u] - G0a[t,s]*G0a[p,u],
                                G0a[p,q]*G0a[t,u] - G0a[t,q]*G0a[p,u],
                                G0a[p,q]*G0a[t,s] - G0a[t,q]*G0a[p,s],
                                G0a[p,s]*G0a[r,u] - G0a[r,s]*G0a[p,u],
                                G0a[p,q]*G0a[r,u] - G0a[r,q]*G0a[p,u],
                                G0a[p,q]*G0a[r,s] - G0a[r,q]*G0a[p,s]]
                    cont3 += cphaseab * numpy.dot(Laa[q,p], Lbb[w,v]) * cofactor[0]
                    cont3 -= cphaseab * numpy.dot(Laa[s,p], Lbb[w,v]) * cofactor[1]
                    cont3 += cphaseab * numpy.dot(Laa[u,p], Lbb[w,v]) * cofactor[2]
                    cont3 -= cphaseab * numpy.dot(Laa[q,r], Lbb[w,v]) * cofactor[3]
                    cont3 += cphaseab * numpy.dot(Laa[s,r], Lbb[w,v]) * cofactor[4]
                    cont3 -= cphaseab * numpy.dot(Laa[u,r], Lbb[w,v]) * cofactor[5]
                    cont3 += cphaseab * numpy.dot(Laa[q,t], Lbb[w,v]) * cofactor[6]
                    cont3 -= cphaseab * numpy.dot(Laa[s,t], Lbb[w,v]) * cofactor[7]
                    cont3 += cphaseab * numpy.dot(Laa[u,t], Lbb[w,v]) * cofactor[8]
                elif (nex_a == 1 and nex_b == 3):
                    p = trial.cre_b[jdet][0]
                    q = trial.anh_b[jdet][0]
                    r = trial.cre_b[jdet][1]
                    s = trial.anh_b[jdet][1]
                    t = trial.cre_b[jdet][2]
                    u = trial.anh_b[jdet][2]
                    v = trial.cre_a[jdet][0]
                    w = trial.anh_a[jdet][0]
                    cofactor = [G0b[r,s]*G0b[t,u] - G0b[r,u]*G0b[t,s],
                               G0b[r,q]*G0b[t,u] - G0b[r,u]*G0b[t,q],
                               G0b[r,q]*G0b[t,s] - G0b[r,s]*G0b[t,q],
                               G0b[p,s]*G0b[t,u] - G0b[t,s]*G0b[p,u],
                               G0b[p,q]*G0b[t,u] - G0b[t,q]*G0b[p,u],
                               G0b[p,q]*G0b[t,s] - G0b[t,q]*G0b[p,s],
                               G0b[p,s]*G0b[r,u] - G0b[r,s]*G0b[p,u],
                               G0b[p,q]*G0b[r,u] - G0b[r,q]*G0b[p,u],
                               G0b[p,q]*G0b[r,s] - G0b[r,q]*G0b[p,s]]

                    cont3 += cphaseab * numpy.dot(Lbb[q,p],Laa[w,v]) * cofactor[0]
                    cont3 -= cphaseab * numpy.dot(Lbb[s,p],Laa[w,v]) * cofactor[1]
                    cont3 += cphaseab * numpy.dot(Lbb[u,p],Laa[w,v]) * cofactor[2]
                    cont3 -= cphaseab * numpy.dot(Lbb[q,r],Laa[w,v]) * cofactor[3]
                    cont3 += cphaseab * numpy.dot(Lbb[s,r],Laa[w,v]) * cofactor[4]
                    cont3 -= cphaseab * numpy.dot(Lbb[u,r],Laa[w,v]) * cofactor[5]
                    cont3 += cphaseab * numpy.dot(Lbb[q,t],Laa[w,v]) * cofactor[6]
                    cont3 -= cphaseab * numpy.dot(Lbb[s,t],Laa[w,v]) * cofactor[7]
                    cont3 += cphaseab * numpy.dot(Lbb[u,t],Laa[w,v]) * cofactor[8]

                else:
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
                                    const = cphaseab * det_cofactor_a * det_cofactor_b
                                    cont3 += numpy.dot(Laa[q,p,:], Lbb[s,r,:]) * const

            if (nex_a == 2): # 4-leg same spin block aaaa
                p = trial.cre_a[jdet][0]
                q = trial.anh_a[jdet][0]
                r = trial.cre_a[jdet][1]
                s = trial.anh_a[jdet][1]
                const = cphasea * ovlpb
                cont3 += (numpy.dot(Laa[q,p,:],Laa[s,r,:])-numpy.dot(Laa[q,r,:],Laa[s,p,:])) * const
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
                                const = cphasea * det_cofactor * ovlpb
                                cont3 +=  (numpy.dot(Laa[q,p,:],Laa[s,r,:])-numpy.dot(Laa[q,r,:],Laa[s,p,:])) * const

            if (nex_b == 2): # 4-leg same spin block bbbb
                p = trial.cre_b[jdet][0]
                q = trial.anh_b[jdet][0]
                r = trial.cre_b[jdet][1]
                s = trial.anh_b[jdet][1]
                const = cphaseb * ovlpa
                cont3 += (numpy.dot(Lbb[q,p,:],Lbb[s,r,:])-numpy.dot(Lbb[q,r,:],Lbb[s,p,:])) * const

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
                                const = cphaseb * det_cofactor * ovlpa
                                cont3 +=  (numpy.dot(Lbb[q,p,:],Lbb[s,r,:])-numpy.dot(Lbb[q,r,:],Lbb[s,p,:])) * const
        cont3 *= (ovlp0/ovlp)

        e2bs += [cont1 + cont2_J + cont2_K + cont3]

    e2bs = numpy.array(e2bs, dtype=numpy.complex128)

    etot = e1bs + e2bs

    energy = numpy.zeros((walker_batch.nwalkers,3),dtype=numpy.complex128)
    energy[:,0] = etot
    energy[:,1] = e1bs
    energy[:,2] = e2bs
    return energy

def get_same_spin_double_contribution_batched(
        cre,
        anh,
        buffer,
        excit_map,
        chol_fact):
    p = cre[:,0]
    q = anh[:,0]
    r = cre[:,1]
    s = anh[:,1]
    contribution = chol_fact[:,q,p]*chol_fact[:,s,r] - chol_fact[:,q,r]*chol_fact[:,s,p]
    buffer[:, excit_map[2]] = contribution

def get_same_spin_double_contribution_batched_contr(
        cre,
        anh,
        buffer,
        excit_map,
        chol_fact):
    p = cre[:,0]
    q = anh[:,0]
    r = cre[:,1]
    s = anh[:,1]
    contribution = numpy.einsum('wJx,wJx->wJ', chol_fact[:,q,p],
            chol_fact[:,s,r], optimize=True)
    contribution -= numpy.einsum('wJx,wJx->wJ', chol_fact[:,q,r],
            chol_fact[:,s,p], optimize=True)
    buffer[:, excit_map[2]] = contribution

def fill_same_spin_contribution_batched(
        nexcit,
        cre,
        anh,
        dets_mat,
        cofactor_matrix,
        chol_fact,
        buffer,
        excit_map
        ):
    nwalkers = dets_mat.shape[0]
    ndet_level = dets_mat.shape[1]
    accumulator = numpy.zeros((nwalkers, ndet_level), dtype=numpy.complex128)
    for iex in range(nexcit):
        for jex in range(nexcit):
            p = cre[:, iex]
            q = anh[:, jex]
            for kex in range(iex+1,nexcit):
                for lex in range(jex+1,nexcit):
                    r = cre[:, kex]
                    s = anh[:, lex]
                    get_cofactor_matrix_4_batched(
                                    nwalkers,
                                    ndet_level,
                                    nexcit,
                                    iex,
                                    jex,
                                    kex,
                                    lex,
                                    dets_mat,
                                    cofactor_matrix)
                    det_cofactor = (-1)**(kex+lex+iex+jex) * numpy.linalg.det(cofactor_matrix)
                    # Nw*Nd
                    dets_contrib = (chol_fact[:,q,p]*chol_fact[:,s,r] - chol_fact[:,q,r]*chol_fact[:,s,p]) * det_cofactor
                    accumulator += dets_contrib

    buffer[:, excit_map[nexcit]] = accumulator

def fill_same_spin_contribution_batched_contr(
        nexcit,
        cre,
        anh,
        dets_mat,
        cofactor_matrix,
        chol_fact,
        buffer,
        excit_map
        ):
    nwalkers = dets_mat.shape[0]
    ndet_level = dets_mat.shape[1]
    accumulator = numpy.zeros((nwalkers, ndet_level), dtype=numpy.complex128)
    t_cof = 0.0
    t_det = 0.0
    t_slice = 0.0
    t_eins = 0.0
    for iex in range(nexcit):
        for jex in range(nexcit):
            p = cre[:, iex]
            q = anh[:, jex]
            chol_a = chol_fact[:,q,p]
            for kex in range(iex+1,nexcit):
                r = cre[:, kex]
                chol_c = chol_fact[:,q,r]
                for lex in range(jex+1,nexcit):
                    s = anh[:, lex]
                    get_cofactor_matrix_4_batched(
                                    nwalkers,
                                    ndet_level,
                                    nexcit,
                                    iex,
                                    jex,
                                    kex,
                                    lex,
                                    dets_mat,
                                    cofactor_matrix)
                    det_cofactor = (-1)**(kex+lex+iex+jex) * numpy.linalg.det(cofactor_matrix)
                    # Nw*Nd
                    chol_b = chol_fact[:,s,r]
                    chol_d = chol_fact[:,s,p]
                    contribution = numpy.einsum('wJx,wJx->wJ', chol_a, chol_b, optimize=True)
                    contribution -= numpy.einsum('wJx,wJx->wJ', chol_c, chol_d, optimize=True)
                    # contribution = numpy.einsum('wJx,wJx->wJ', chol_fact[:,q,p], chol_fact[:,s,r], optimize=True)
                    # contribution -= numpy.einsum('wJx,wJx->wJ', chol_fact[:,q,r], chol_fact[:,s,p], optimize=True)
                    contribution *= det_cofactor
                    accumulator += contribution

    buffer[:, excit_map[nexcit]] = accumulator

def fill_opp_spin_factors_batched(
        nexcit,
        cre,
        anh,
        excit_map,
        det_matrix,
        cofactor_matrix,
        chol_factor,
        spin_buffer):
    nwalkers = cofactor_matrix.shape[0]
    ndet_level = cofactor_matrix.shape[1]
    accumulator = numpy.zeros((nwalkers, ndet_level), dtype=numpy.complex128)
    for iex in range(nexcit):
        ps = cre[:, iex]
        for jex in range(nexcit):
            qs = anh[:, jex]
            get_cofactor_matrix_batched(
                    nwalkers,
                    ndet_level,
                    nexcit,
                    iex,
                    jex,
                    det_matrix,
                    cofactor_matrix)
            # Nw x Nd_l
            det_cofactors = (-1)**(iex+jex) * numpy.linalg.det(cofactor_matrix)
            accumulator += det_cofactors * chol_factor[:, qs, ps]
    spin_buffer[:, excit_map[nexcit]] = accumulator

def fill_opp_spin_factors_batched_chol(
        nexcit,
        cre,
        anh,
        excit_map,
        det_matrix,
        cofactor_matrix,
        chol_factor,
        spin_buffer):
    nwalkers = cofactor_matrix.shape[0]
    ndet_level = cofactor_matrix.shape[1]
    accumulator = numpy.zeros((nwalkers, ndet_level, chol_factor.shape[-1]), dtype=numpy.complex128)
    for iex in range(nexcit):
        ps = cre[:, iex]
        for jex in range(nexcit):
            qs = anh[:, jex]
            get_cofactor_matrix_batched(
                    nwalkers,
                    ndet_level,
                    nexcit,
                    iex,
                    jex,
                    det_matrix,
                    cofactor_matrix)
            # Nw x Nd_l
            det_cofactors = (-1)**(iex+jex) * numpy.linalg.det(cofactor_matrix)
            accumulator += det_cofactors[:,:,None] * chol_factor[:, qs, ps]
    spin_buffer[:, excit_map[nexcit]] = accumulator

def fill_opp_spin_factors_batched_singles(
        cre,
        anh,
        excit_map,
        chol_factor,
        spin_buffer):
    p = cre[:, 0]
    q = anh[:, 0]
    spin_buffer[:, excit_map[1]] = chol_factor[:, q, p]

def fill_opp_spin_factors_batched_doubles(
        cre,
        anh,
        excit_map,
        G0,
        chol_factor,
        spin_buffer):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    accumulator  = chol_factor[:,q,p] * G0[:,r,s]
    accumulator -= chol_factor[:,s,p] * G0[:,r,q]
    accumulator -= chol_factor[:,q,r] * G0[:,p,s]
    accumulator += chol_factor[:,s,r] * G0[:,p,q]
    spin_buffer[:, excit_map[2]] = accumulator

def fill_opp_spin_factors_batched_doubles_chol(
        cre,
        anh,
        excit_map,
        G0,
        chol_factor,
        spin_buffer):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    accumulator  = chol_factor[:,q,p] * G0[:,r,s,None]
    accumulator -= chol_factor[:,s,p] * G0[:,r,q,None]
    accumulator -= chol_factor[:,q,r] * G0[:,p,s,None]
    accumulator += chol_factor[:,s,r] * G0[:,p,q,None]
    spin_buffer[:, excit_map[2]] = accumulator

def fill_opp_spin_factors_batched_triples(
        cre,
        anh,
        excit_map,
        G0,
        chol_factor,
        spin_buffer):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    t = cre[:, 2]
    u = anh[:, 2]
    accumulator  = chol_factor[:,q,p] * (
                        G0[:,r,s]*G0[:,t,u] - G0[:,r,u]*G0[:,t,s]
                        )
    accumulator -= chol_factor[:,s,p] * (
                        G0[:,r,q]*G0[:,t,u] - G0[:,r,u]*G0[:,t,q]
                        )
    accumulator += chol_factor[:,u,p] * (
                        G0[:,r,q]*G0[:,t,s] - G0[:,r,s]*G0[:,t,q]
                        )
    accumulator -= chol_factor[:,q,r] * (
                        G0[:,p,s]*G0[:,t,u] - G0[:,t,s]*G0[:,p,u]
                        )
    accumulator += chol_factor[:,s,r] * (
                        G0[:,p,q]*G0[:,t,u] - G0[:,t,q]*G0[:,p,u]
                        )
    accumulator -= chol_factor[:,u,r] * (
                        G0[:,p,q]*G0[:,t,s] - G0[:,t,q]*G0[:,p,s]
                        )
    accumulator += chol_factor[:,q,t] * (
                        G0[:,p,s]*G0[:,r,u] - G0[:,r,s]*G0[:,p,u]
                        )
    accumulator -= chol_factor[:,s,t] * (
                        G0[:,p,q]*G0[:,r,u] - G0[:,r,q]*G0[:,p,u]
                        )
    accumulator += chol_factor[:,u,t] * (
                        G0[:,p,q]*G0[:,r,s] - G0[:,r,q]*G0[:,p,s]
                        )
    spin_buffer[:, excit_map[3]] = accumulator

def fill_opp_spin_factors_batched_triples_chol(
        cre,
        anh,
        excit_map,
        G0,
        chol_factor,
        spin_buffer):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    t = cre[:, 2]
    u = anh[:, 2]
    cofac = G0[:,r,s]*G0[:,t,u] - G0[:,r,u]*G0[:,t,s]
    accumulator  = chol_factor[:,q,p] * cofac[:,:,None]
    cofac = G0[:,r,q]*G0[:,t,u] - G0[:,r,u]*G0[:,t,q]
    accumulator -= chol_factor[:,s,p] * cofac[:,:,None]
    cofac = G0[:,r,q]*G0[:,t,s] - G0[:,r,s]*G0[:,t,q]
    accumulator += chol_factor[:,u,p] * cofac[:,:,None]
    cofac = G0[:,p,s]*G0[:,t,u] - G0[:,t,s]*G0[:,p,u]
    accumulator -= chol_factor[:,q,r] * cofac[:,:,None]
    cofac = G0[:,p,q]*G0[:,t,u] - G0[:,t,q]*G0[:,p,u]
    accumulator += chol_factor[:,s,r] * cofac[:,:,None]
    cofac = G0[:,p,q]*G0[:,t,s] - G0[:,t,q]*G0[:,p,s]
    accumulator -= chol_factor[:,u,r] * cofac[:,:,None]
    cofac = G0[:,p,s]*G0[:,r,u] - G0[:,r,s]*G0[:,p,u]
    accumulator += chol_factor[:,q,t] * cofac[:,:,None]
    cofac = G0[:,p,q]*G0[:,r,u] - G0[:,r,q]*G0[:,p,u]
    accumulator -= chol_factor[:,s,t] * cofac[:,:,None]
    cofac = G0[:,p,q]*G0[:,r,s] - G0[:,r,q]*G0[:,p,s]
    accumulator += chol_factor[:,u,t] * cofac[:,:,None]
    spin_buffer[:, excit_map[3]] = accumulator

def local_energy_multi_det_trial_wicks_batch_opt_low_mem(system, ham, walker_batch, trial, iw=None):
    assert iw == None
    nwalkers = walker_batch.nwalkers
    nbasis = ham.nbasis
    nchol = ham.nchol
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis*nbasis))
    Gb = walker_batch.Gb.reshape((nwalkers, nbasis*nbasis))
    e1b = Ga.dot(ham.H1[0].ravel()) + Gb.dot(ham.H1[1].ravel()) + ham.ecore

    ovlpa0 = walker_batch.det_ovlpas[:,0]
    ovlpb0 = walker_batch.det_ovlpbs[:,0]
    ovlp0 = ovlpa0 * ovlpb0
    ovlp = walker_batch.ot

    # useful variables
    G0a = walker_batch.G0a
    G0b = walker_batch.G0b
    Q0a = walker_batch.Q0a
    Q0b = walker_batch.Q0b
    CIa = walker_batch.CIa
    CIb = walker_batch.CIb

    # contribution 1 (disconnected)
    Xa = numpy.dot(G0a.reshape((-1, nbasis*nbasis)), ham.chol_vecs)
    Xb = numpy.dot(G0b.reshape((-1, nbasis*nbasis)), ham.chol_vecs)
    ecoul = numpy.einsum("wx,wx->w", Xa, Xa, optimize=True)
    ecoul += numpy.einsum("wx,wx->w", Xb, Xb, optimize=True)
    ecoul += 2. * numpy.einsum("wx,wx->w", Xa, Xb, optimize=True)

    Ta = numpy.zeros((nwalkers, nbasis, nbasis), dtype=numpy.complex128)
    Tb = numpy.zeros((nwalkers, nbasis, nbasis), dtype=numpy.complex128)
    exx  = numpy.zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(nchol):  # write a cython function that calls blas for this.
        chol_x = ham.chol_vecs[:,x].reshape((nbasis, nbasis))
        Ta[:,:,:] = numpy.dot(G0a, chol_x.T)
        Tb[:,:,:] = numpy.dot(G0b, chol_x.T)
        exx += (
                numpy.einsum("wpr,wrp->w", Ta, Ta, optimize=True) +
                numpy.einsum("wpr,wrp->w", Tb, Tb, optimize=True)
                )

    cont1 = 0.5 * (ecoul - exx)
    P0 = G0a + G0b

    LXa = numpy.einsum("wx,mx->wm", Xa, ham.chol_vecs, optimize=True)
    LXb = numpy.einsum("wx,mx->wm", Xb, ham.chol_vecs, optimize=True)
    LXa = LXa.reshape((nwalkers, nbasis, nbasis))
    LXb = LXb.reshape((nwalkers, nbasis, nbasis))

    # useful intermediate
    QCIGa = numpy.einsum(
                'wpr,wrs,wsq->wpq',
                walker_batch.Q0a,
                walker_batch.CIa,
                G0a,
                optimize=True)
    QCIGb = numpy.einsum(
                'wpr,wrs,wsq->wpq',
                walker_batch.Q0b,
                walker_batch.CIb,
                G0b,
                optimize=True)

    cont2_Jaa = numpy.einsum('wpq,wpq->w', QCIGa, LXa, optimize=True)
    cont2_Jbb = numpy.einsum('wpq,wpq->w', QCIGb, LXb, optimize=True)
    cont2_Jab = numpy.einsum('wpq,wpq->w', QCIGa, LXb, optimize=True)
    cont2_Jba = numpy.einsum('wpq,wpq->w', QCIGb, LXa, optimize=True)
    cont2_J = cont2_Jaa + cont2_Jbb + cont2_Jab + cont2_Jba
    cont2_J *= (ovlp0/ovlp)

    # Second, Exchange-like term
    cont2_Kaa = numpy.zeros(nwalkers, dtype=numpy.complex128)
    cont2_Kbb = numpy.zeros(nwalkers, dtype=numpy.complex128)
    na = walker_batch.nup
    nb = walker_batch.ndown
    for x in range(nchol):
        Lmn = ham.chol_vecs[:,x].reshape((nbasis, nbasis))
        LGL = numpy.einsum('pr,wsr,sq->wpq', Lmn, G0a, Lmn, optimize=True) # merge this with energy 0 eval
        cont2_Kaa -= numpy.einsum('wpq,wpq->w', LGL, QCIGa, optimize=True)

        LGL = numpy.einsum('pr,wsr,sq->wpq', Lmn, G0b, Lmn, optimize=True)
        cont2_Kaa -= numpy.einsum('wpq,wpq->w', LGL, QCIGb, optimize=True)

    cont2_Kaa *= (ovlp0/ovlp)
    cont2_Kbb *= (ovlp0/ovlp)

    cont2_K = cont2_Kaa + cont2_Kbb
    dets_a_full, dets_b_full = compute_determinants_batched(G0a, G0b, trial)

    ndets = len(trial.coeffs)
    chol_vecs = ham.chol_vecs.reshape((nbasis, nbasis, nchol))
    energy_os = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    energy_ss = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    cphase_a = trial.coeffs.conj() * trial.phase_a
    cphase_b = trial.coeffs.conj() * trial.phase_b
    ovlpa = dets_a_full * trial.phase_a[None,:]
    ovlpb = dets_b_full * trial.phase_b[None,:]
    c_phasea_ovlpb = cphase_a[None,:] * ovlpb
    c_phaseb_ovlpa = cphase_b[None,:] * ovlpa
    cphase_ab = cphase_a * trial.phase_b
    for ix in range(nchol):
        Lx = chol_vecs[:,:,ix]
        Laa = numpy.einsum('wiq,wpj,ij->wqp', Q0a, G0a, Lx, optimize=True)
        Lbb = numpy.einsum('wiq,wpj,ij->wqp', Q0b, G0b, Lx, optimize=True)
        alpha_os_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        beta_os_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        alpha_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        beta_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        for iexcit in range(1, trial.max_excite+1):
            ndets_a = len(trial.cre_ex_a[iexcit])
            det_mat_a = numpy.zeros((nwalkers, ndets_a, iexcit, iexcit), dtype=numpy.complex128)
            get_det_matrix_batched(
                    iexcit,
                    trial.cre_ex_a[iexcit],
                    trial.anh_ex_a[iexcit],
                    walker_batch.G0a,
                    det_mat_a)
            # defined here for reuse
            cofactor_matrix_a = numpy.zeros((nwalkers, ndets_a, max(iexcit-1,1), max(iexcit-1,1)), dtype=numpy.complex128)
            if ndets_a > 0:
                if iexcit == 1:
                    fill_opp_spin_factors_batched_singles(
                            trial.cre_ex_a[iexcit],
                            trial.anh_ex_a[iexcit],
                            trial.excit_map_a,
                            Laa,
                            alpha_os_buffer
                            )
                elif iexcit == 2:
                    fill_opp_spin_factors_batched_doubles(
                            trial.cre_ex_a[iexcit],
                            trial.anh_ex_a[iexcit],
                            trial.excit_map_a,
                            G0a,
                            Laa,
                            alpha_os_buffer
                            )
                elif iexcit == 3:
                    fill_opp_spin_factors_batched_triples(
                            trial.cre_ex_a[iexcit],
                            trial.anh_ex_a[iexcit],
                            trial.excit_map_a,
                            G0a,
                            Laa,
                            alpha_os_buffer
                            )
                else:
                    fill_opp_spin_factors_batched(
                                    iexcit,
                                    trial.cre_ex_a[iexcit],
                                    trial.anh_ex_a[iexcit],
                                    trial.excit_map_a,
                                    det_mat_a,
                                    cofactor_matrix_a,
                                    Laa,
                                    alpha_os_buffer)
            if iexcit >= 2 and ndets_a > 0:
                if iexcit == 2:
                    get_same_spin_double_contribution_batched(
                            trial.cre_ex_a[iexcit],
                            trial.anh_ex_a[iexcit],
                            alpha_ss_buffer,
                            trial.excit_map_a,
                            Laa)
                else:
                    fill_same_spin_contribution_batched(
                                        iexcit,
                                        trial.cre_ex_a[iexcit],
                                        trial.anh_ex_a[iexcit],
                                        det_mat_a,
                                        cofactor_matrix_a[:,:,:max(iexcit-2,1),:max(iexcit-2,1)],
                                        Laa,
                                        alpha_ss_buffer,
                                        trial.excit_map_a
                                        )
            ndets_b = len(trial.cre_ex_b[iexcit])
            det_mat_b = numpy.zeros((nwalkers, ndets_b, iexcit, iexcit), dtype=numpy.complex128)
            get_det_matrix_batched(
                    iexcit,
                    trial.cre_ex_b[iexcit],
                    trial.anh_ex_b[iexcit],
                    walker_batch.G0b,
                    det_mat_b)
            cofactor_matrix_b = numpy.zeros((nwalkers, ndets_b, max(iexcit-1,1), max(iexcit-1,1)), dtype=numpy.complex128)
            if ndets_b > 0:
                if iexcit == 1:
                    fill_opp_spin_factors_batched_singles(
                            trial.cre_ex_b[iexcit],
                            trial.anh_ex_b[iexcit],
                            trial.excit_map_b,
                            Lbb,
                            beta_os_buffer
                            )
                elif iexcit == 2:
                    fill_opp_spin_factors_batched_doubles(
                            trial.cre_ex_b[iexcit],
                            trial.anh_ex_b[iexcit],
                            trial.excit_map_b,
                            G0b,
                            Lbb,
                            beta_os_buffer
                            )
                elif iexcit == 3:
                    fill_opp_spin_factors_batched_triples(
                            trial.cre_ex_b[iexcit],
                            trial.anh_ex_b[iexcit],
                            trial.excit_map_b,
                            G0b,
                            Lbb,
                            beta_os_buffer
                            )
                else:
                    fill_opp_spin_factors_batched(
                                    iexcit,
                                    trial.cre_ex_b[iexcit],
                                    trial.anh_ex_b[iexcit],
                                    trial.excit_map_b,
                                    det_mat_b,
                                    cofactor_matrix_b,
                                    Lbb,
                                    beta_os_buffer)
            if iexcit >= 2 and ndets_b > 0:
                if iexcit == 2:
                    get_same_spin_double_contribution_batched(
                            trial.cre_ex_b[iexcit],
                            trial.anh_ex_b[iexcit],
                            beta_ss_buffer,
                            trial.excit_map_b,
                            Lbb)
                else:
                    fill_same_spin_contribution_batched(
                                        iexcit,
                                        trial.cre_ex_b[iexcit],
                                        trial.anh_ex_b[iexcit],
                                        det_mat_b,
                                        cofactor_matrix_b[:,:,:max(iexcit-2,1),:max(iexcit-2,1)],
                                        Lbb,
                                        beta_ss_buffer,
                                        trial.excit_map_b
                                        )

        # accumulating over x (cholesky vector)
        energy_os += numpy.einsum('wJ,J->wJ', alpha_os_buffer*beta_os_buffer, cphase_ab, optimize=True)
        energy_ss += numpy.einsum('wJ,wJ->wJ', alpha_ss_buffer, c_phasea_ovlpb, optimize=True)
        energy_ss += numpy.einsum('wJ,wJ->wJ', beta_ss_buffer, c_phaseb_ovlpa, optimize=True)

    cont3 = (numpy.sum(energy_os + energy_ss, axis=1))*(ovlp0/ovlp)
    e2b = cont1 + cont2_J + cont2_K + cont3

    walker_energies = numpy.zeros((nwalkers, 3), dtype=numpy.complex128)
    walker_energies[:,0] = e1b + e2b
    walker_energies[:,1] = e1b
    walker_energies[:,2] = e2b
    return walker_energies

def local_energy_multi_det_trial_wicks_batch_opt(system, ham, walker_batch, trial, iw=None):
    import time
    start = time.time()
    assert iw == None
    nwalkers = walker_batch.nwalkers
    nbasis = ham.nbasis
    nchol = ham.nchol
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis*nbasis))
    Gb = walker_batch.Gb.reshape((nwalkers, nbasis*nbasis))
    e1b = Ga.dot(ham.H1[0].ravel()) + Gb.dot(ham.H1[1].ravel()) + ham.ecore

    ovlpa0 = walker_batch.det_ovlpas[:,0]
    ovlpb0 = walker_batch.det_ovlpbs[:,0]
    ovlp0 = ovlpa0 * ovlpb0
    ovlp = walker_batch.ot

    # useful variables
    G0a = walker_batch.G0a
    G0b = walker_batch.G0b
    Q0a = walker_batch.Q0a
    Q0b = walker_batch.Q0b
    CIa = walker_batch.CIa
    CIb = walker_batch.CIb

    # contribution 1 (disconnected)
    Xa = numpy.dot(G0a.reshape((-1, nbasis*nbasis)), ham.chol_vecs)
    Xb = numpy.dot(G0b.reshape((-1, nbasis*nbasis)), ham.chol_vecs)
    cont1 = local_energy_single_det_batch_einsum(system, ham, walker_batch, trial)[:,2]
    P0 = G0a + G0b

    LXa = numpy.einsum("wx,mx->wm", Xa, ham.chol_vecs, optimize=True)
    LXb = numpy.einsum("wx,mx->wm", Xb, ham.chol_vecs, optimize=True)
    LXa = LXa.reshape((nwalkers, nbasis, nbasis))
    LXb = LXb.reshape((nwalkers, nbasis, nbasis))

    # useful intermediate
    QCIGa = numpy.einsum(
                'wpr,wrs,wsq->wpq',
                walker_batch.Q0a,
                walker_batch.CIa,
                G0a,
                optimize=True)
    QCIGb = numpy.einsum(
                'wpr,wrs,wsq->wpq',
                walker_batch.Q0b,
                walker_batch.CIb,
                G0b,
                optimize=True)

    cont2_Jaa = numpy.einsum('wpq,wpq->w', QCIGa, LXa, optimize=True)
    cont2_Jbb = numpy.einsum('wpq,wpq->w', QCIGb, LXb, optimize=True)
    cont2_Jab = numpy.einsum('wpq,wpq->w', QCIGa, LXb, optimize=True)
    cont2_Jba = numpy.einsum('wpq,wpq->w', QCIGb, LXa, optimize=True)
    cont2_J = cont2_Jaa + cont2_Jbb + cont2_Jab + cont2_Jba
    cont2_J *= (ovlp0/ovlp)

    # Second, Exchange-like term
    cont2_Kaa = numpy.zeros(nwalkers, dtype=numpy.complex128)
    cont2_Kbb = numpy.zeros(nwalkers, dtype=numpy.complex128)
    na = walker_batch.nup
    nb = walker_batch.ndown
    for x in range(nchol):
        Lmn = ham.chol_vecs[:,x].reshape((nbasis, nbasis))
        # LGL = numpy.einsum('wsr,pr,sq->wpq', G0a, Lmn, Lmn, optimize=True)
        LGL = [(Lmn @ G0a[iwalker].T) @ Lmn for iwalker in range(nwalkers)]
        cont2_Kaa -= numpy.einsum('wpq,wpq->w', LGL, QCIGa, optimize=True)

        # LGL = numpy.einsum('wsr,pr,sq->wpq', G0b, Lmn, Lmn, optimize=True)
        # LGL = Lmn.dot(G0b.T).T.dot(Lmn)
        LGL = [(Lmn @ G0b[iwalker].T) @ Lmn for iwalker in range(nwalkers)]
        cont2_Kaa -= numpy.einsum('wpq,wpq->w', LGL, QCIGb, optimize=True)

    cont2_Kaa *= (ovlp0/ovlp)
    cont2_Kbb *= (ovlp0/ovlp)

    cont2_K = cont2_Kaa + cont2_Kbb

    # Laa = numpy.einsum('wiq,wpj,xij->wqpx', Q0a, G0a, chol_vecs, optimize=True)
    # Lbb = numpy.einsum('wpj,wqjx->wqpx', G0a, T, optimize=True)
    Laa = numpy.zeros((nwalkers, nbasis, nbasis, nchol), dtype=numpy.complex128)
    Lbb = numpy.zeros((nwalkers, nbasis, nbasis, nchol), dtype=numpy.complex128)
    # This is **much** faster than the above einsum
    chol_vecs = ham.chol_vecs.reshape((nbasis, nbasis, nchol))
    # start2 = time.time()
    for i in range(nwalkers):
        Laa[i] = numpy.einsum("iq,pj,ijx->qpx", Q0a[i], G0a[i], chol_vecs, optimize=True)
        Lbb[i] = numpy.einsum("iq,pj,ijx->qpx", Q0b[i], G0b[i], chol_vecs, optimize=True)
    cont3 = 0.0
    # print(time.time()-start)
    dets_a_full, dets_b_full = compute_determinants_batched(G0a, G0b, trial)
    ndets = len(trial.coeffs)
    energy_os = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    energy_ss = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    cphase_a = trial.coeffs.conj() * trial.phase_a
    cphase_b = trial.coeffs.conj() * trial.phase_b
    ovlpa = dets_a_full * trial.phase_a[None,:]
    ovlpb = dets_b_full * trial.phase_b[None,:]
    c_phasea_ovlpb = cphase_a[None,:] * ovlpb
    c_phaseb_ovlpa = cphase_b[None,:] * ovlpa
    cphase_ab = cphase_a * trial.phase_b
    alpha_os_buffer = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    beta_os_buffer = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    alpha_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    beta_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    for iexcit in range(1, trial.max_excite+1):
        ndets_a = len(trial.cre_ex_a[iexcit])
        det_mat_a = numpy.zeros((nwalkers, ndets_a, iexcit, iexcit), dtype=numpy.complex128)
        # defined here for reuse
        cofactor_matrix_a = numpy.zeros((nwalkers, ndets_a, max(iexcit-1,1), max(iexcit-1,1)), dtype=numpy.complex128)
        get_det_matrix_batched(
                iexcit,
                trial.cre_ex_a[iexcit],
                trial.anh_ex_a[iexcit],
                walker_batch.G0a,
                det_mat_a)
        if ndets_a > 0:
            if iexcit == 1:
                fill_opp_spin_factors_batched_singles(
                        trial.cre_ex_a[iexcit],
                        trial.anh_ex_a[iexcit],
                        trial.excit_map_a,
                        Laa,
                        alpha_os_buffer
                        )
            elif iexcit == 2:
                fill_opp_spin_factors_batched_doubles_chol(
                        trial.cre_ex_a[iexcit],
                        trial.anh_ex_a[iexcit],
                        trial.excit_map_a,
                        G0a,
                        Laa,
                        alpha_os_buffer
                        )
            elif iexcit == 3:
                fill_opp_spin_factors_batched_triples_chol(
                        trial.cre_ex_a[iexcit],
                        trial.anh_ex_a[iexcit],
                        trial.excit_map_a,
                        G0a,
                        Laa,
                        alpha_os_buffer
                        )
            else:
                fill_opp_spin_factors_batched_chol(
                                iexcit,
                                trial.cre_ex_a[iexcit],
                                trial.anh_ex_a[iexcit],
                                trial.excit_map_a,
                                det_mat_a,
                                cofactor_matrix_a,
                                Laa,
                                alpha_os_buffer)
            if iexcit >= 2 and ndets_a > 0:
                if iexcit == 2:
                    get_same_spin_double_contribution_batched_contr(
                            trial.cre_ex_a[iexcit],
                            trial.anh_ex_a[iexcit],
                            alpha_ss_buffer,
                            trial.excit_map_a,
                            Laa)
                else:
                    fill_same_spin_contribution_batched_contr(
                                        iexcit,
                                        trial.cre_ex_a[iexcit],
                                        trial.anh_ex_a[iexcit],
                                        det_mat_a,
                                        cofactor_matrix_a[:,:,:max(iexcit-2,1),:max(iexcit-2,1)],
                                        Laa,
                                        alpha_ss_buffer,
                                        trial.excit_map_a
                                        )
        ndets_b = len(trial.cre_ex_b[iexcit])
        det_mat_b = numpy.zeros((nwalkers, ndets_b, iexcit, iexcit), dtype=numpy.complex128)
        cofactor_matrix_b = numpy.zeros((nwalkers, ndets_b, max(iexcit-1,1), max(iexcit-1,1)), dtype=numpy.complex128)
        get_det_matrix_batched(
                iexcit,
                trial.cre_ex_b[iexcit],
                trial.anh_ex_b[iexcit],
                walker_batch.G0b,
                det_mat_b)
        if ndets_b > 0:
            # if iexcit > 2:
            if iexcit == 1:
                fill_opp_spin_factors_batched_singles(
                        trial.cre_ex_b[iexcit],
                        trial.anh_ex_b[iexcit],
                        trial.excit_map_b,
                        Lbb,
                        beta_os_buffer
                        )
            elif iexcit == 2:
                fill_opp_spin_factors_batched_doubles_chol(
                        trial.cre_ex_b[iexcit],
                        trial.anh_ex_b[iexcit],
                        trial.excit_map_b,
                        G0b,
                        Lbb,
                        beta_os_buffer
                        )
            elif iexcit == 3:
                fill_opp_spin_factors_batched_triples_chol(
                        trial.cre_ex_b[iexcit],
                        trial.anh_ex_b[iexcit],
                        trial.excit_map_b,
                        G0b,
                        Lbb,
                        beta_os_buffer
                        )
            else:
                fill_opp_spin_factors_batched_chol(
                                iexcit,
                                trial.cre_ex_b[iexcit],
                                trial.anh_ex_b[iexcit],
                                trial.excit_map_b,
                                det_mat_b,
                                cofactor_matrix_b,
                                Lbb,
                                beta_os_buffer)
        if iexcit >= 2 and ndets_b > 0:
            if iexcit == 2:
                get_same_spin_double_contribution_batched_contr(
                        trial.cre_ex_b[iexcit],
                        trial.anh_ex_b[iexcit],
                        beta_ss_buffer,
                        trial.excit_map_b,
                        Lbb)
            else:
                fill_same_spin_contribution_batched_contr(
                                    iexcit,
                                    trial.cre_ex_b[iexcit],
                                    trial.anh_ex_b[iexcit],
                                    det_mat_b,
                                    cofactor_matrix_b[:,:,:max(iexcit-2,1),:max(iexcit-2,1)],
                                    Lbb,
                                    beta_ss_buffer,
                                    trial.excit_map_b
                                    )

    # accumulating over x (cholesky vector)
    energy_os = numpy.einsum('wJx,wJx,J->w', alpha_os_buffer, beta_os_buffer, cphase_ab, optimize=True)
    energy_ss = numpy.einsum('wJ,wJ->w', alpha_ss_buffer, c_phasea_ovlpb, optimize=True)
    energy_ss += numpy.einsum('wJ,wJ->w', beta_ss_buffer, c_phaseb_ovlpa, optimize=True)

    cont3 = (energy_os + energy_ss)*(ovlp0/ovlp)
    e2b = cont1 + cont2_J + cont2_K + cont3

    walker_energies = numpy.zeros((nwalkers, 3), dtype=numpy.complex128)
    walker_energies[:,0] = e1b + e2b
    walker_energies[:,1] = e1b
    walker_energies[:,2] = e2b
    return walker_energies


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
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.T)

    ecoul = einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * einsum("xw,xw->w", Xa, Xb, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    # GhalfaT_batch = walker_batch.Ghalfa.transpose(0,2,1).copy() # nw x nbasis x nocc
    # GhalfbT_batch = walker_batch.Ghalfb.transpose(0,2,1).copy() # nw x nbasis x nocc

    Ta = zeros((nwalkers, nalpha,nalpha), dtype=numpy.complex128)
    Tb = zeros((nwalkers, nbeta,nbeta), dtype=numpy.complex128)

    exx  = zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    # breakpoint()
    for x in range(nchol):  # write a cython function that calls blas for this.
        rmi_a = trial._rchola[x].reshape((nalpha,nbasis))
        rmi_b = trial._rcholb[x].reshape((nbeta,nbasis))
        # if (isrealobj(trial._rchola)):
            # Ta += rmi_a.dot(GhalfaT_batch.real).transpose(1,0,2) + 1j * rmi_a.dot(GhalfaT_batch.imag).transpose(1,0,2)
            # Tb += rmi_b.dot(GhalfbT_batch.real).transpose(1,0,2) + 1j * rmi_b.dot(GhalfbT_batch.imag).transpose(1,0,2)
        # else:
            # Ta += rmi_a.dot(GhalfaT_batch).transpose(1,0,2)
            # Tb += rmi_b.dot(GhalfbT_batch).transpose(1,0,2)
        Ta = walker_batch.Ghalfa @ rmi_a.T
        Tb = walker_batch.Ghalfb @ rmi_b.T

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
