# Sampling rate and noise

## Sampling rate

@Doval2006: source-filter theory only valid until 4 or 5 kHz. Formant tables usually go up to 5500 Hz, so settle for 5 kHz. This gives a sampling interval of 0.1 msec, or about 70 samples per pitch period assuming $T$ = 150 Hz.

## Data normalization and the nugget term

Given the data vector $d_i, i = 1 \ldots I$, we normalize to unit power:
$$\sum_{i=1}^I d_i^2/I := 1$$

This has many advantages over the conventional normalization, where the maximum amplitude is chosen to be one, because we know that the SNR for our problem is at about 20 dB or higher from earlier experiments and from LPC [@Schroeder1999]. The SNR of the physical signals on the interval $[0,T]$ is defined as
$$SNR = \frac{\int_0^T |d(t)|^2 dt}{\int_0^T |e(t)|^2 dt}$$

where $d(t)$ and $e(t) = d(t) - f(t)$ are the underlying continuous data and error signals. Given the discrete data $d_i$ we can approximate:
$$\int |d(t)|^2 dt \approx \sum_{i=1}^I d_i^2 \Delta$$

where $\Delta = 1/f_s$. And if we model the residual as white noise of amplitude power $\sigma_n^2$ with a GP, we have
$$\int |e(t)|^2 dt = \sigma_n^2 T = \sigma_n^2 \Delta I$$

Thus the SNR reduces to
$$SNR = \frac{\sum_{i=1}^I d_i^2 \Delta}{\sigma_n^2 \Delta I} = \sigma_n^{-2}$$

by virtue of our normalization policy, and we know that the SNR for our problem is about 0.01 = 20 dB or more; thus we can immediately put a meaningful prior over $\sigma_n^2$ because we know that $\sigma_n^2 \approx 0.01$ or smaller. Note that the SNR definition is independent of which normalization is used; what we accomplish is setting a convenient scale for $\sigma_n^2$ and the model amplitudes by fixing the reference value in our dB log scale to unity.

Indeed, on this scale of $d_i$ we also know that the noise residuals have a worst-case mean absolute amplitude of about $\sqrt{0.01} = 0.1$, though this will typically be half or much less of that number.[^pareto] And we know that the total noise power on the interval $[0,T]$ is about $0.01 T$, which we use to set upper bounds for the smoothness factor $r = \ell/T$. The rationale for this is that we can discard signals with significantly less power than the expected noise.

[^pareto]: I confirmed this with the Pareto chain experiments, where the SNR is 20 dB worst-case and typically 25 dB or more.

This also sets the scale for the marginal variances of GPs at three levels.

- At the "signal" level, i.e. a signal $f(t) \sim GP(0,k)$ with comparable power to the data to be modeled, we have that
  $$\langle \int_0^T |f(t)|^2 dt \rangle = \int_0^T k(t,t) = \sigma^2 T$$
  where the last equation is due to assumed stationarity of $f(t)$ and $\sigma^2$ is the marginal variance. But our normalization policy also implies
  $$\langle \int_0^T |f(t)|^2 dt \rangle \approx \sum_{i=1}^I d_i^2 \Delta = T$$
  thus $\sigma^2 \approx 1$.
- At the "noise" level, we already saw that $\sigma_n^2 \lessapprox 0.01$, or a SNR of 20 dB or better.
- At the "nugget" level, this also sets a useful scale for the nugget term $\delta^2$ necessary for the stability of inverting the GP kernels. It must be anything very small compared to 0.01, but large enough to stabilize Cholesky decomposition. We set the noise floor at -60 dB below the expected $\sigma_n^2 = -20 dB$ such that $\delta^2 = 10^{-8}$, such that $k(t,t) \rightarrow k(t,t) + \delta^2$. This can also be understood as an informative prior for the amplitudes with have zero-mean independent Gaussians with variance $\sigma_n^2/\delta^2$ [@Bretthorst1991, Eq. 25].