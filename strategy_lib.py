import random
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist


def chol_update(L, a12, a22, lower=True):
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


def imse(X_cand, gp, X_ref):
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


def pimse(X_cand, pgp, X_ref, global_search=True):
    C_cand = pgp.region_classifier.predict(X_cand)
    C_ref = pgp.region_classifier.predict(X_ref)
    cand_labels = np.unique(C_cand)
    ref_labels = np.unique(C_ref)
    score = np.zeros(len(X_cand))

    sub_var_vals = np.zeros(len(pgp.local_gp))
    for c in ref_labels:         
        sub_var_vals[c] = max(np.square(pgp.local_gp[c].predict(X_ref[C_ref == c], return_std=True)[1]).mean(), 0)

    if global_search:
        c_sel = np.argmax(stats.multinomial(1, sub_var_vals / sub_var_vals.sum()).rvs())
        
        if c_sel not in C_cand:
            raise Exception("No candidate is not in the most uncertain region")
        else:
            pass
        
        idx_sel_cand = np.where(C_cand == c_sel)[0]
        idx_sel_ref = np.where(C_ref == c_sel)[0]
        score[idx_sel_cand] = imse(X_cand[idx_sel_cand], pgp.local_gp[c_sel], X_ref[idx_sel_ref])
        return score

    else:
        score = np.zeros(len(X_cand))
        for c in cand_labels:
            score[C_cand == c] = -imse(X_cand[C_cand == c], pgp.local_gp[c], X_ref[C_ref == c]) \
                                      + sub_var_vals.sum()
        return score


def physcal_integrate(X_cand, f_gp, h_gp, xi, weight=0.5, p=1, safety=0.95, ref_safety=0.9, alpha=2., X_ref=None, prod=False):
    # check ref_safety is less than safety
    assert ref_safety < safety, "ref_safety should be less than safety"

    mu_h, sigma_h = h_gp.predict(X_cand, return_std=True)
    safe_idx = np.where(mu_h + stats.norm.ppf(safety) * sigma_h < xi)
    S = X_cand[safe_idx]

    if X_ref is None:
        X_ref = X_cand.copy()
        mu_h_ref, sigma_h_ref = mu_h.copy(), sigma_h.copy()
    else:
        mu_h_ref, sigma_h_ref = h_gp.predict(X_ref, return_std=True)

    S_ref = X_ref[np.where(mu_h_ref + stats.norm.ppf(ref_safety) * sigma_h_ref < xi)]

    # safe variance reduction
    safe_variance_reduction_score = imse(S, f_gp, S_ref)

    # safe region expansion
    mu_h_safe, sigma_h_safe = mu_h[safe_idx], sigma_h[safe_idx]
    eta = alpha * sigma_h_safe
    safe_region_expansion_score = np.square(eta) * (stats.norm.cdf(xi, mu_h_safe, sigma_h_safe) - stats.norm.cdf(xi - eta, mu_h_safe, sigma_h_safe))
    for i in range(len(mu_h_safe)):
        safe_region_expansion_score[i] -= stats.norm.expect(lambda x: np.square(x - xi), loc=mu_h_safe[i], scale=sigma_h_safe[i], lb=xi - eta[i], ub=xi)

    score_mat = np.column_stack((safe_variance_reduction_score, safe_region_expansion_score))
    score_divisor = score_mat.max(0) - score_mat.min(0)
    normalized_score_mat = np.divide((score_mat - score_mat.min(0)), score_divisor, out=np.zeros_like(score_mat), where=score_divisor != 0)

    score = np.zeros(len(X_cand))
    score_mat = np.zeros((len(X_cand), 2))
    score_mat[safe_idx, :] = normalized_score_mat

    weight_vec = np.array([1 - weight, weight])
    if not prod:
        score[safe_idx] = np.dot(normalized_score_mat ** p, weight_vec) ** (1 / p)
    else:
        score[safe_idx] = np.prod(normalized_score_mat, axis=1)
    return score, score_mat


def segp(X_cand, f_gp, h_gp, safety=0.95):
    def nuisance_value(x, h_gp):
        K_ = h_gp.base_estimator_.kernel_(h_gp.base_estimator_.X_train_, x)
        mu_ = K_.T.dot(h_gp.base_estimator_.y_train_ - h_gp.base_estimator_.pi_)
        v_ = solve_triangular(h_gp.base_estimator_.L_, h_gp.base_estimator_.W_sr_[:, np.newaxis] * K_, lower=True)
        sigma_ = np.sqrt(h_gp.base_estimator_.kernel_.diag(x) - np.einsum("ij,ij->j", v_, v_))
        return mu_, sigma_

    mu_h, sigma_h = nuisance_value(X_cand, h_gp)
    safe_idx = np.where(mu_h + stats.norm.ppf(safety) * sigma_h >= 0)
    S = X_cand[safe_idx]
    
    score = np.zeros(len(X_cand))
    score[safe_idx] = max_entropy(S, f_gp)
    return score



### Below would be deprecated
# def safe_region_expansion(X, gp, xi, alpha=2., truncate=True):
#     mu, sigma = gp.predict(X, return_std=True)
#     mu = mu.ravel()
#     eps = alpha * sigma
#     if truncate:
#         eif_val = np.square(eps) * (stats.norm.cdf(xi, mu, sigma) - stats.norm.cdf(xi - eps, mu, sigma)) - int_vfun_(mu, xi, sigma, alpha, truncate)
#     else:
#         eif_val = (np.square(eps) - np.square(mu - xi)) * (stats.norm.cdf(xi + eps, mu, sigma) - stats.norm.cdf(xi - eps, mu, sigma)) \
#                   + 2 * (mu - xi) * np.square(sigma) * (stats.norm.pdf(xi + eps, mu, sigma) - stats.norm.pdf(xi - eps, mu, sigma)) \
#                   - int_vfun_(mu, xi, sigma, alpha, truncate)
#     return eif_val


# def eff(X, gp, xi, alpha=2.):
#     mu, sigma = gp.predict(X, return_std=True)
#     mu = mu.ravel()
#     eps = alpha * sigma
#     xi_ub = xi + eps
#     xi_lb = xi - eps
#     eff_val = (mu - xi) * (2 * stats.norm.cdf(xi, mu, sigma) - stats.norm.cdf(xi_lb, mu, sigma) - stats.norm.cdf(xi_ub, mu, sigma)) \
#               - sigma * (2 * stats.norm.pdf(xi, mu, sigma) - stats.norm.pdf(xi_lb, mu, sigma) - stats.norm.pdf(xi_ub, mu, sigma)) \
#               + eps * (stats.norm.cdf(xi_ub, mu, sigma) - stats.norm.cdf(xi_lb, mu, sigma))
#     return eff_val


# def safe_var_reduction(X_cand, f_gp, h_gp, xi, safety=0.95, ref_safety=0.9, X_ref=None):
#     score = np.zeros(len(X_cand))
#     score[safe_idx] = imse(S, f_gp, S_ref)
#     return score

# def vfun_(h, xi, mu, sigma):
#     return np.square(h - xi) * stats.norm.pdf(h, mu, sigma)


# def int_vfun_(mu, xi, sigma, alpha, truncate=True):
#     vals = np.zeros_like(mu)
#     ub = xi + alpha * sigma
#     lb = xi - alpha * sigma
#     for i in range(len(mu)):
#         if truncate:
#             vals[i] = quad(vfun_, lb[i], xi, args=(xi, mu[i], sigma[i]))[0]
#         else:
#             vals[i] = quad(vfun_, lb[i], ub[i], args=(xi, mu[i], sigma[i]))[0]
#     return vals
