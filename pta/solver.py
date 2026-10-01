"""The closed-form transfer: a two-sided ridge solve of the descriptor-matching objective.

A descriptor U (N x M) is matched on a grid of N probe embeddings F (N x d) and M
prompt landmarks G (M x d) of the target. The edit f -> (I + alpha A) f changes the
first-order affinities by alpha F A^T G^T, so the operator is found from

    X* = argmin_X ||F X G^T - U||^2 + mu_G ||F X||^2 + mu_F ||X G^T||^2 + mu_F mu_G ||X||^2,
    A  = X*^T,

with mu_F = ridge_f * mean(sigma(F)^2) and mu_G = ridge_g * mean(sigma(G)^2). With
the thin SVDs F = U_F S_F V_F^T and G^T = U_G S_G V_G^T the minimizer is

    X* = V_F diag(s / (s^2 + mu_F)) U_F^T  U  V_G diag(t / (t^2 + mu_G)) U_G^T,

so the map U -> X* is linear on a fixed grid (Proposition 1) and a bank of
operators solved on one grid composes by addition exactly.
"""
import torch


def ridge_solver(F, G, ridge_f=1e-2, ridge_g=1e-6):
    """Factor the target side of one grid once and return `solve(U) -> X`.

    F: (N, d) probe embeddings of the target, G: (M, d) prompt landmarks.
    The returned X has shape (d, d) and is the operator's transpose, so that
    row features are edited as f + alpha * f @ X. Everything is computed in
    double precision and returned in F's dtype.
    """
    U_F, S_F, Vh_F = torch.linalg.svd(F.double(), full_matrices=False)
    filt_F = S_F / (S_F ** 2 + ridge_f * float((S_F ** 2).mean()))
    U_G, S_G, Vh_G = torch.linalg.svd(G.t().double(), full_matrices=False)
    filt_G = S_G / (S_G ** 2 + ridge_g * float((S_G ** 2).mean()))
    left = Vh_F.t() * filt_F
    right = (U_G * filt_G).t()
    U_Ft, V_G, dtype = U_F.t(), Vh_G.t(), F.dtype

    def solve(U):
        return (left @ (U_Ft @ U.double() @ V_G) @ right).to(dtype)
    return solve


def ridge_solve(F, G, U, ridge_f=1e-2, ridge_g=1e-6):
    """One-shot form of `ridge_solver`."""
    return ridge_solver(F, G, ridge_f, ridge_g)(U)


def rank_capped_solve(F, G, U, rank, ridge_f=1e-2, ridge_g=1e-6):
    """The same objective under rank(X) <= rank, in closed form.

    Substituting Z = S_F V_F^T X U_G S_G reduces the fit term to ||Z - U_F^T U V_G||^2,
    and rank(X) = rank(Z), so the constrained problem is Eckart-Young on
    U_F^T U V_G, with the ridge entering through the same filters. Used by the
    rank ablation only; the paper's operators are solved at full rank.
    """
    U_F, S_F, Vh_F = torch.linalg.svd(F.double(), full_matrices=False)
    filt_F = S_F / (S_F ** 2 + ridge_f * float((S_F ** 2).mean()))
    U_G, S_G, Vh_G = torch.linalg.svd(G.t().double(), full_matrices=False)
    filt_G = S_G / (S_G ** 2 + ridge_g * float((S_G ** 2).mean()))
    M0 = U_F.t() @ U.double() @ Vh_G.t()
    Uz, Sz, Vzh = torch.linalg.svd(M0, full_matrices=False)
    k = min(rank, Sz.numel())
    Z = (Uz[:, :k] * Sz[:k]) @ Vzh[:k]
    return ((Vh_F.t() * filt_F) @ Z @ (U_G * filt_G).t()).to(F.dtype)


def relative_residual(F, G, X, U):
    """||F X G^T - U|| / ||U||, the part of the descriptor the solve leaves unmatched."""
    return float(((F @ X) @ G.t() - U).norm() / U.norm())
