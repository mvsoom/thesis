from gpjax.kernels import AbstractKernel


class AbstractStationaryKernel(AbstractKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stationary = True
