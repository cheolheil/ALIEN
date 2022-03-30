import random
import numpy as np
from scipy.stats import norm, multinomial
from scipy.integrate import quad
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist


def chol_update(L, a12, a22, lower=True):
    # check triangular
    if lower:
        if np.all(np.triu(L, k=1) == 0.):
            pass
        else:
            raise Exception("L is not a lower triangle matrix")
    else:
        if np.all(np.tril(L, k=1) == 0.):
            L = L.T
        else:
            raise Exception("L is not an upper triangle matrix")

    # shape check
    n = len(L)
    if len(a12) != n:
        raise Exception("a12 length must be n")
    elif np.ndim(a12) != 2:
        a12 = a12[:, np.newaxis]
    else:
        pass

    l12 = solve_triangular(L, a12, lower=True)
    l22 = np.sqrt(a22 - np.square(l12).sum())
    L_sol = np.vstack((
        np.hstack((L, np.zeros((n, 1)))),
        np.hstack((l12.T, l22))
    ))
    return L_sol


def max_entropy(X_cand, gp):
    score = np.square(gp.predict(X_cand, return_std=True)[1])
    return score


def max_distance(X_cand, gp):
    score = cdist(X_cand, gp.X_train_).min(axis=1)
    return score


def random_sample(X_cand, gp):
    score = np.arange(len(X_cand))
    random.shuffle(score)
    return score


def fast_imse(X_cand, gp, X_ref):
    n = len(X_cand)
    m = len(X_ref)
    score = np.zeros(n)
    k_ref = gp.kernel_(gp.X_train_, X_ref)
    L = gp.L_
    v = solve_triangular(L, k_ref, lower=True)
    for i in range(n):
        xi = X_cand[i][np.newaxis, :]
        k12 = gp.kernel_(gp.X_train_, xi)
        k22 = gp.kernel_(xi) + 1e-10
        k_ = gp.kernel_(xi, X_ref)
        L_ = chol_update(L, k12, k22)
        l12 = L_[-1, :-1]
        k_ -= l12.reshape(1, -1) @ v
        score[i] = np.inner(k_, k_) / m
    return score


def fast_pimse(X_cand, pgp, X_ref, global_search=True):
    C_cand = pgp.region_classifier.predict(X_cand)
    C_ref = pgp.region_classifier.predict(X_ref)
    cand_labels = np.unique(C_cand)
    ref_labels = np.unique(C_ref)
    score = np.zeros(len(X_cand))

    sub_var_vals = np.zeros(len(pgp.local_gp))
    for c in ref_labels:
        # if c in pgp.unknown_classes:
        #     sub_var_vals[c] = np.inf
        # else:            
        sub_var_vals[c] = max(np.square(pgp.local_gp[c].predict(X_ref[C_ref == c], return_std=True)[1]).mean(), 0)

    if global_search:
        # choose the most uncertain region`
        c_sel = np.argmax(multinomial(1, sub_var_vals / sub_var_vals.sum()).rvs())
        
        if c_sel not in C_cand:
            raise Exception("No candidate is not in the most uncertain region")
        else:
            pass
        
        # save sample indices of the most uncertain region
        idx_sel_cand = np.where(C_cand == c_sel)[0]
        # if c_sel in pgp.unknown_classes:
        #     score[np.random.choice(idx_sel_cand)] = 1
        #     return score
        # else:
        idx_sel_ref = np.where(C_ref == c_sel)[0]
        score[idx_sel_cand] = fast_imse(X_cand[idx_sel_cand], pgp.local_gp[c_sel], X_ref[idx_sel_ref])
        return score

    else:
        score = np.zeros(len(X_cand))
        for c in cand_labels:
            # if c in pgp.unknown_classes:
            #     score[C_cand == c] = -np.inf
            # else:
            score[C_cand == c] = -fast_imse(X_cand[C_cand == c], pgp.local_gp[c], X_ref[C_ref == c]) \
                                      + sub_var_vals.sum()
        return score
