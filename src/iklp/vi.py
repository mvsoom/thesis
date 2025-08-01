# %%

from __future__ import annotations

import jax
import jax.numpy as jnp

from .gig import gig_dkl_from_gamma
from .hyperparams import random_periodic_kernel_hyperparams
from .mercer_op import logdet, solve, solve_normal_eq, trinv, trinv_Ki
from .psi import psi_matvec
from .state import (
    Auxiliaries,
    VIState,
    compute_auxiliaries,
    compute_expectations,
    init_test_stable_state,
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


def update_theta(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)

    alpha = state.data.h.alpha
    quads = (
        (1 / aux.E.nu_w_inv)
        * ((1 / aux.E.theta_inv) ** 2)
        * compute_unscaled_quads(state, aux)
    )

    new_rho_theta = 2 * alpha + aux.E.nu_w * trinv_Ki(aux.Omega)
    new_tau_theta = quads

    new_xi = state.xi.replace(
        rho_theta=new_rho_theta,
        tau_theta=new_tau_theta,
    )

    return state.replace(xi=new_xi)


def update_nu_w(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)

    bw = state.data.h.bw
    quads = ((1 / aux.E.nu_w_inv) ** 2) * compute_unscaled_quads(state, aux)

    new_rho_w = 2 * bw + jnp.dot(aux.E.theta, trinv_Ki(aux.Omega))
    new_tau_w = jnp.dot(1 / aux.E.theta_inv, quads)

    new_xi = state.xi.replace(
        rho_w=new_rho_w,
        tau_w=new_tau_w,
    )

    return state.replace(xi=new_xi)


def update_nu_e(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)

    be = state.data.h.be
    quad0 = ((1 / aux.E.nu_e_inv) ** 2) * compute_unscaled_quad0(state, aux)

    new_rho_e = 2 * be + trinv(aux.Omega)
    new_tau_e = quad0

    new_xi = state.xi.replace(
        rho_e=new_rho_e,
        tau_e=new_tau_e,
    )

    return state.replace(xi=new_xi)


def update_delta_a(state: VIState) -> VIState:
    aux = compute_auxiliaries(state)

    lam = state.data.h.lam

    # Solve normal equation with S operator, as Sigma^(-1) == S^(-1)
    new_delta_a = solve_normal_eq(aux.S, lam)

    new_xi = state.xi.replace(
        delta_a=new_delta_a,
    )

    return state.replace(xi=new_xi)


def compute_elbo_bound(state: VIState):
    aux = compute_auxiliaries(state)

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
            aux.E.nu_w * jnp.sum(aux.E.theta * trinv_Ki(aux.Omega)),
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
    delta_a_term = (
        0.5
        * (1 / state.data.h.lam)
        * jnp.dot(state.xi.delta_a, state.xi.delta_a)
    )

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


# @partial(jax.jit, donate_argnums=(0,))  # TODO: if we record state, do not donate argnum
def vi_step(state: VIState) -> VIState:
    state = update_theta(state)
    state = update_nu_w(state)
    state = update_nu_e(state)
    state = update_delta_a(state)
    return state


def update_next(state: VIState, i) -> VIState:
    if i % 4 == 0:
        state = update_theta(state)
    elif i % 4 == 1:
        state = update_nu_w(state)
    elif i % 4 == 2:
        state = update_nu_e(state)
    elif i % 4 == 3:
        state = update_delta_a(state)
    return state


def vi_run(
    state: VIState, data: Data, compute_elbo=False, record=False
) -> VIState:
    # Updating q(a) = delta(a* - a) as the very first update
    # is known to yield better convergence
    # as it is initalized to zeroes
    state = update_delta_a(state)

    def scan_fn(state, batch):
        new_state, _ = vi_step(state, batch)
        # return new carry, and record the new carry as output
        return new_state, new_state

    final_state, states = jax.lax.scan(scan_fn, init_state, batches)
    # states has shape [T, *state_tree_shapes]
    return final_state, states


def vi_test(state: VIState) -> VIState:
    state = update_theta(state)
    state = update_nu_w(state)
    state = update_nu_e(state)
    state = update_delta_a(state)
    return state


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    key = jax.random.PRNGKey(123)

    def sk():
        global key
        key, k = jax.random.split(key)
        return k

    # Jit stuff
    if True:
        update_delta_a = jax.jit(update_delta_a)
        vi_step = jax.jit(vi_step)
        compute_elbo_bound = jax.jit(compute_elbo_bound)
        vi_step_test = jax.jit(vi_step_test)
        update_next = jax.jit(update_next, static_argnames=["i"])

    # %%
    # Test (first run very slow, then fast)
    h = random_periodic_kernel_hyperparams(
        sk(), I=400, M=512, hyper_kwargs={"P": 12}
    )
    state = init_test_stable_state(sk(), h)

    import matplotlib.pyplot as plt

    plt.figure()
    cmap = plt.get_cmap("coolwarm")

    # Updating q(a) = delta(a* - a) as the very first update
    # is known to yield better convergence
    # as it is initalized to zeroes
    state = update_delta_a(state)

    score = -jnp.inf
    criterion = 0.0001
    n_iter = 20

    for i in range(n_iter):
        state = vi_step(state)

        E = compute_expectations(state)
        color = cmap(i / n_iter)
        plt.plot(E.theta, color=color, alpha=0.8, label=f"iter {i}")

        lastscore = score
        score = compute_elbo_bound(state)

        if i == 0:
            improvement = 1.0
        else:
            improvement = (score - lastscore) / jnp.abs(lastscore)

        print(
            "iteration {}: bound = {:.2f} ({:+.5f} improvement)".format(
                i, score, improvement
            )
        )
        if improvement < 0.0:
            print("Diverged")
            break
        if improvement < criterion:
            print("Converged")
            break
        if jnp.isnan(improvement) and i > 0:
            print("NaN")
            break

    plt.legend()
    plt.xlabel("i")
    plt.ylabel("theta")
    plt.yscale("log")
