# %%
import gpjax as gpx
import jax.numpy as jnp
from gpjax.kernels.computations import DenseKernelComputation


class gpxBlockIndependent(gpx.kernels.AbstractKernel):
    """
    Block-diagonal kernel enforcing independence across trajectories.

    Inputs are 1D vectors of length 2:
        x = [traj_id, tau]
        y = [traj_id', tau']

    The kernel is defined as:
        k((n, tau), (n', tau')) =
            k_time(tau, tau')    if n == n'
            0                    otherwise

    This yields an exact block-diagonal covariance matrix when data from
    multiple trajectories are concatenated, while sharing kernel
    hyperparameters and inducing points.
    """

    def __init__(self, base_kernel):
        super().__init__(None, None, DenseKernelComputation())
        self.base_kernel = base_kernel

    def __call__(self, x, y):
        """Dispatch on shape. Inducing points are not bound by the trajectory_index"""
        # Case 1: both are inducing points (shape (1,))
        if x.shape[-1] == 1 and y.shape[-1] == 1:
            return self.base_kernel(x, y).squeeze()

        # Case 2: inducing (1,) vs data (2,)
        if x.shape[-1] == 1 and y.shape[-1] == 2:
            tau_y = y[..., 1:]
            return self.base_kernel(x, tau_y).squeeze()

        if x.shape[-1] == 2 and y.shape[-1] == 1:
            tau_x = x[..., 1:]
            return self.base_kernel(tau_x, y).squeeze()

        # Case 3: data vs data (block diagonal)
        id_x, tau_x = x[..., 0], x[..., 1:]
        id_y, tau_y = y[..., 0], y[..., 1:]

        k_time = self.base_kernel(tau_x, tau_y)
        return jnp.where(id_x == id_y, k_time, 0.0).squeeze()


if __name__ == "__main__":
    from prism.pack import gpxPACK

    k = gpxPACK(d=1)
    kb = gpxBlockIndependent(k)

    x = jnp.array([[0, 1.45]])
    y = jnp.array([[1, 2.56]])

    print("Equal (want: False)?", kb(x, y) == k(x[0, 1], y[0, 1]))
    print("Equal to zero? (want: True)", kb(x, y) == 0.0)
