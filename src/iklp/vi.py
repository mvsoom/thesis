from __future__ import annotations

import jax
import jax.numpy as jnp

from .gig import gig_dkl_from_gamma
from .mercer_op import logdet, solve, solve_normal_eq, trinv
from .psi import psi_matvec
from .state import (
    Auxiliaries,
    VIState,
    compute_auxiliaries,
)


def solve_w(state: VIState, aux: Auxiliaries):
    e = psi_matvec(state.xi.delta_a, state.data.x)  # (M,)
    w = solve(aux.S, e)  # (M,)
    return w  # (M,)


def compute_unscaled_quads(state: VIState, aux: Auxiliaries, w=None):
    if w is None:
        w = solve_w(state, aux)

    p = jnp.einsum("IMr,M->Ir", state.data.h.Phi, w)  # (I, r)
    unscaled_quads = jnp.einsum("Ir,Ir->I", p, p)  # (I,)

    return unscaled_quads  # (I,)


def compute_unscaled_quad0(state: VIState, aux: Auxiliaries, w=None):
    if w is None:
        w = solve_w(state, aux)

    unscaled_quad0 = jnp.dot(w, w)  # ()
    return unscaled_quad0  # ()


def update_theta_aux(state: VIState, aux: Auxiliaries) -> VIState:
    alpha = state.data.h.alpha
    quads = (
        (1 / aux.E.nu_w_inv)
        * ((1 / aux.E.theta_inv) ** 2)
        * compute_unscaled_quads(state, aux)
    )

    new_rho_theta = 2 * alpha + aux.E.nu_w * aux.tKi_Omega
    new_tau_theta = quads

    new_xi = state.xi.replace(
        rho_theta=new_rho_theta,
        tau_theta=new_tau_theta,
    )

    return state.replace(xi=new_xi)


def update_theta(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)
    return update_theta_aux(state, aux)


def update_nu_w_aux(state: VIState, aux: Auxiliaries) -> VIState:
    bw = state.data.h.bw
    quads = ((1 / aux.E.nu_w_inv) ** 2) * compute_unscaled_quads(state, aux)

    new_rho_w = 2 * bw + jnp.dot(aux.E.theta, aux.tKi_Omega)
    new_tau_w = jnp.dot(1 / aux.E.theta_inv, quads)

    new_xi = state.xi.replace(
        rho_w=new_rho_w,
        tau_w=new_tau_w,
    )

    return state.replace(xi=new_xi)


def update_nu_w(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)
    return update_nu_w_aux(state, aux)


def update_nu_e_aux(state: VIState, aux: Auxiliaries) -> VIState:
    be = state.data.h.be
    quad0 = ((1 / aux.E.nu_e_inv) ** 2) * compute_unscaled_quad0(state, aux)

    new_rho_e = 2 * be + trinv(aux.Omega)
    new_tau_e = quad0

    new_xi = state.xi.replace(
        rho_e=new_rho_e,
        tau_e=new_tau_e,
    )

    return state.replace(xi=new_xi)


def update_nu_e(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)
    return update_nu_e_aux(state, aux)


def update_delta_a_aux(state: VIState, aux: Auxiliaries) -> VIState:
    # Solve normal equation with S operator, as Sigma^(-1) == S^(-1)
    new_delta_a = solve_normal_eq(aux.S, state.data.h.arprior)

    new_xi = state.xi.replace(
        delta_a=new_delta_a,
    )

    return state.replace(xi=new_xi)


def update_delta_a(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)
    return update_delta_a_aux(state, aux)


def compute_elbo_bound(state: VIState):
    aux = compute_auxiliaries(state)
    return compute_elbo_bound_aux(state, aux)


def compute_elbo_bound_aux(state: VIState, aux: Auxiliaries):
    # Compute likelihood terms (eq 23)
    w = solve_w(state, aux)

    quads = (
        (1 / aux.E.nu_w_inv)
        * (1 / aux.E.theta_inv)
        * compute_unscaled_quads(state, aux, w)
    )
    quad0 = (1 / aux.E.nu_e_inv) * compute_unscaled_quad0(state, aux, w)

    likelihood_bound_terms = jnp.asarray(
        [
            logdet(aux.Omega),
            aux.E.nu_w * jnp.sum(aux.E.theta * aux.tKi_Omega),
            aux.E.nu_e * trinv(aux.Omega),
            jnp.sum(quads),
            quad0,
        ]
    )

    # jax.debug.print("Likelihood terms: {}", likelihood_bound_terms)

    likelihood_bound = -0.5 * jnp.sum(likelihood_bound_terms)

    # Compute D_KL terms (eq 19)
    I = state.data.h.Phi.shape[0]

    theta_term = jnp.sum(
        jax.vmap(gig_dkl_from_gamma, in_axes=(0, 0, 0, 0, None, None))(
            aux.E.theta,  # (I,)
            aux.E.theta_inv,  # (I,)
            state.xi.rho_theta,  # (I,)
            state.xi.tau_theta,  # (I,)
            state.data.h.alpha / I,  # ()
            state.data.h.alpha,  # ()
        )
    )

    nu_w_term = gig_dkl_from_gamma(
        aux.E.nu_w,
        aux.E.nu_w_inv,
        state.xi.rho_w,
        state.xi.tau_w,
        state.data.h.aw,
        state.data.h.bw,
    )

    nu_e_term = gig_dkl_from_gamma(
        aux.E.nu_e,
        aux.E.nu_e_inv,
        state.xi.rho_e,
        state.xi.tau_e,
        state.data.h.ae,
        state.data.h.be,
    )

    # delta_a_term = D_KL( delta | MVN ) with diverging part and constants removed -- makes sense if we assume delta is a narrow epsilon-MVN and discard all constants depending on hyperparameters (lambda and epsilon)
    # It doesnt affect optimization of delta_a per se, but it does reflect the regularization normal equation (function of lambda) used for solving for a
    # plus it affects the bound convergence, so indirectly the number of iterations etc.
    # Note: MacKay (2005) does the same, and the D_KL term *is* included in Yoshii, and it is a common VI strategy, so we do the same
    arprior = state.data.h.arprior
    d = state.xi.delta_a - arprior.mean
    delta_a_term = 0.5 * d @ (arprior.precision @ d)

    kl_terms = jnp.asarray(
        [
            theta_term,  # >= 0
            nu_w_term,  # >= 0
            nu_e_term,  # >= 0
            delta_a_term,  # >= 0
        ]
    )

    # jax.debug.print("KL terms: {}", kl_terms)

    kl_term = jnp.sum(kl_terms)

    # Add together: ELBO_bound = E_q(bounded likelihood) - D_KL(q|prior)
    elbo_bound = likelihood_bound - kl_term
    return elbo_bound


def vi_step_test(state: VIState) -> VIState:
    jax.debug.print("ELBO bound: {}", compute_elbo_bound(state))
    state = update_theta(state)
    jax.debug.print(
        "ELBO bound after update_theta: {}", compute_elbo_bound(state)
    )
    state = update_nu_w(state)
    jax.debug.print(
        "ELBO bound after update_nu_w: {}", compute_elbo_bound(state)
    )
    state = update_nu_e(state)
    jax.debug.print(
        "ELBO bound after update_nu_e: {}", compute_elbo_bound(state)
    )
    state = update_delta_a(state)
    jax.debug.print(
        "ELBO bound after update_delta_a: {}", compute_elbo_bound(state)
    )
    return state


def vi_step_debug(state: VIState) -> VIState:
    # NOTE: No need for donate_argnums here.
    # vi_step() runs inside a jitted lax.scan, so the scan carry (state) is already
    # input-output aliased by XLA. The .replace(...) calls in the update_*(state) functions
    # only create new container objects; unchanged leaves (e.g., state.data) keep the same
    # device buffers, and updated leaves can reuse old storage via carry aliasing.
    # donate_argnums at this level would only matter at a real host => device call boundary,
    # which this is not.

    # Updating q(a) = delta(a* - a) as the very first update is known to yield better convergence as it is initalized to zeroes
    state = update_delta_a(state)

    state = update_theta(state)
    state = update_nu_w(state)
    state = update_nu_e(state)

    return state


def vi_step(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)
    state = update_delta_a_aux(state, aux)
    state = update_theta_aux(state, aux)

    aux = compute_auxiliaries(state)
    state = update_nu_w_aux(state, aux)

    aux = compute_auxiliaries(state)
    state = update_nu_e_aux(state, aux)

    return state
