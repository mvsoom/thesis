= Parametric, nonparametric, semiparametric?

We clarify some of the confusing terminology associated with Bayesian nonparametric models.

=== Parametric models
are simply models with a fixed and prespecified amount of unknown parameters $R$, independent of the size of the dataset $N$.

Examples are linear regression and neural networks.
In BNGIF the filter priors $pi_"PZ"$ and $pi_"AP"$ priors are parametric as they have a fixed amount of parameters $R$ depending on their order $K$, similar to linear prediction (LP) methods.
And of course the $pi_"LF"$ prior with $R = 4$ is parametric as well. // TODO

=== Nonparametric models
in a Bayesian context are models for which the effective number
of parameters $tilde(R)$ is capable of adapting to $N$ @Orbanz2010.

They are also referred to as infinite-dimensional parametric models, which is just a shorthand way of saying that they are parametric models which have been mathematically constructed to behave consistently under inference as $R -> oo$.
That is, given the data of size $N$, a nonparametric model roughly behaves as an ordinary parametric model with $tilde(R) < oo$ parameters, with the exception that $tilde(R)$ is determined automatically from the data rather than specified beforehand.

Examples are Dirichlet process mixture models, for which the effective number of mixture components does not need to be known beforehand, and, of course, GPs, which are infinite-dimensional distributions over functions.
// % GPs are NNs with infinite width $R = oo$.
In BNGIF the $pi_"GP"$ and $pi_"VS"$ priors are nonparametric because they are GPs.#footnote[
  Strictly, these are not fully nonparametric because they are only reduced-rank GPs and not full-rank GPs.
  Their capacity to adapt their complexity to growing amounts of data $N$ is limited by the number of basisfunctions $M$;
  nonparametric behavior is recovered in the limit $M -> oo$.
]

// Other nonparametric models are the multi-output GPs for the various parameter trajectories.

The name 'nonparametric' is a bit of a misnomer because nonparametric models almost always have a set of $Q$ hyperparameters which typically determine either the scales present in the data or the degree to which sparsity or smoothness is encouraged.
In theory, inference of these $Q$ hyperparameters is no different from inference with a parametric model of order $R = Q + tilde(R)$ in which $tilde(R)$ parameters have been marginalized over analytically.

// Another confusing is nonparametric in classical statistics, not Bayesian.

=== Semiparametric models
in a Bayesian context are models with both parametric and nonparametric components for which the latter is considered a nuisance parameter @Ghosal2017[p.~368].
This means that we are mainly interested in conclusions about the parametric component of the model, which require marginalizing over the nonparametric component.

In BNGIF the parametric and nonparametric components are very neatly divided into models for the filter $h(t)$ and the source $u(t)$, respectively, reflecting the qualitative difference between the information available from acoustic phonetics about the two.
One might be tempted to call BNGIF semiparametric for this reason if it weren't for the fact that the nonparametric component describing $u(t)$ is actually the main object of interest for the GIF problem.
If anything the nuisance bit for GIF is in the parametric component describing $h(t)$, not the nonparametric component for $u(t)$.
// %In addition, ``the term nonparametric is so popular that it makes little sense not to use it'' \citep[][p.~1]{Ghosal2017}, while Bayesian semiparametric models are more obscure.
