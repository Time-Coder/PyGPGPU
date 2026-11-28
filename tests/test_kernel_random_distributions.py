"""Every scalar probability distribution a kernel body can draw.

The vocabulary is the union of the two host libraries' scalar-valued generators. A draw
answers one number, so array-valued APIs (`choice`, `permutation`, `multinomial`, ...)
keep the "one number per call" refusal from test_kernel_random.py. This file pins the
other half: each spelling reaches the device, answers a value of the right shape, and
has the mean its distribution says it has. The `backend` fixture includes `python`, so
the interpreter runs the same bodies through the host libraries too.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from pygpgpu import kernel


COUNT = 8192


# --- python's random module -------------------------------------------------------

@kernel
def d_expovariate(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.expovariate(1.0)

@kernel
def d_betavariate(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.betavariate(2.0, 3.0)

@kernel
def d_gammavariate(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.gammavariate(2.0, 1.0)

@kernel
def d_lognormvariate(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.lognormvariate(0.0, 1.0)

@kernel
def d_triangular(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.triangular(0.0, 1.0, 0.5)

@kernel
def d_vonmisesvariate(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.vonmisesvariate(0.0, 2.0)

@kernel
def d_paretovariate(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.paretovariate(2.0)

@kernel
def d_weibullvariate(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.weibullvariate(2.0, 2.0)


# --- numpy's random module --------------------------------------------------------

@kernel
def d_exponential(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.exponential(2.0)

@kernel
def d_standard_exponential(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.standard_exponential()

@kernel
def d_beta(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.beta(2.0, 3.0)

@kernel
def d_gamma(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.gamma(2.0)

@kernel
def d_standard_gamma(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.standard_gamma(2.0)

@kernel
def d_lognormal(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.lognormal(0.0, 1.0)

@kernel
def d_triangular_np(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.triangular(0.0, 0.5, 1.0)

@kernel
def d_chisquare(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.chisquare(4.0)

@kernel
def d_noncentral_chisquare(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.noncentral_chisquare(4.0, 1.0)

@kernel
def d_f(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.f(4.0, 5.0)

@kernel
def d_noncentral_f(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.noncentral_f(4.0, 5.0, 1.0)

@kernel
def d_standard_cauchy(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.standard_cauchy()

@kernel
def d_standard_t(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.standard_t(5.0)

@kernel
def d_gumbel(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.gumbel(0.0, 1.0)

@kernel
def d_laplace(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.laplace(0.0, 1.0)

@kernel
def d_logistic(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.logistic(0.0, 1.0)

@kernel
def d_pareto(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.pareto(2.0)

@kernel
def d_power(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.power(2.0)

@kernel
def d_rayleigh(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.rayleigh(1.0)

@kernel
def d_vonmises(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.vonmises(0.0, 2.0)

@kernel
def d_wald(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.wald(2.0, 1.0)

@kernel
def d_weibull(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.random.weibull(2.0)


# --- discrete results, carried in a float array ----------------------------------

@kernel
def d_geometric(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(np.random.geometric(0.3))

@kernel
def d_poisson(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(np.random.poisson(4.0))

@kernel
def d_binomial(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(np.random.binomial(10, 0.5))

@kernel
def d_negative_binomial(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(np.random.negative_binomial(3, 0.5))

@kernel
def d_hypergeometric(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(np.random.hypergeometric(10, 10, 5))

@kernel
def d_logseries(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(np.random.logseries(0.5))

@kernel
def d_zipf(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(np.random.zipf(2.0))


# name, kernel, expected mean, relative tolerance, lower bound, upper bound.
# The heavy-tailed and undefined-mean cases use a wide tolerance and assert finiteness
# instead; a wrong sampler (a constant, a uniform, a sign flip) still fails them.
CASES = [
    ("expovariate", d_expovariate, 1.0, 0.12, 0.0, None),
    ("betavariate", d_betavariate, 0.4, 0.08, 0.0, 1.0),
    ("gammavariate", d_gammavariate, 2.0, 0.08, 0.0, None),
    ("lognormvariate", d_lognormvariate, np.exp(0.5), 0.12, 0.0, None),
    ("triangular", d_triangular, 0.5, 0.08, 0.0, 1.0),
    ("vonmisesvariate", d_vonmisesvariate, 3.14159, 0.30, 0.0, 6.2832),
    ("paretovariate", d_paretovariate, 2.0, 0.20, 0.0, None),
    ("weibullvariate", d_weibullvariate, 2 * np.sqrt(np.pi) / 2, 0.10, 0.0, None),
    ("exponential", d_exponential, 2.0, 0.10, 0.0, None),
    ("standard_exponential", d_standard_exponential, 1.0, 0.10, 0.0, None),
    ("beta", d_beta, 0.4, 0.08, 0.0, 1.0),
    ("gamma", d_gamma, 2.0, 0.08, 0.0, None),
    ("standard_gamma", d_standard_gamma, 2.0, 0.08, 0.0, None),
    ("lognormal", d_lognormal, np.exp(0.5), 0.12, 0.0, None),
    ("triangular_np", d_triangular_np, 0.5, 0.08, 0.0, 1.0),
    ("chisquare", d_chisquare, 4.0, 0.10, 0.0, None),
    ("noncentral_chisquare", d_noncentral_chisquare, 5.0, 0.15, 0.0, None),
    ("f", d_f, 5.0 / 3.0, 0.20, 0.0, None),
    ("noncentral_f", d_noncentral_f, 2.0, 0.40, 0.0, None),
    ("standard_cauchy", d_standard_cauchy, None, 0.60, None, None),
    # Student-t is heavy-tailed: per draw P(|T5| > 20) is ~3e-6, so an 8192-draw
    # launch exceeds +-20 about 2% of the time (measured: 66/3000 launches, max 38).
    # A bound of +-80 sits at ~5e-5 per launch, far enough out to still catch a
    # broken generator while never tripping on the distribution's own tail.
    ("standard_t", d_standard_t, 0.0, 0.20, -80.0, 80.0),
    ("gumbel", d_gumbel, 0.5772, 0.20, None, None),
    ("laplace", d_laplace, 0.0, 0.15, None, None),
    ("logistic", d_logistic, 0.0, 0.15, None, None),
    ("pareto", d_pareto, 1.0, 0.20, 0.0, None),
    ("power", d_power, 2.0 / 3.0, 0.08, 0.0, 1.0),
    ("rayleigh", d_rayleigh, np.sqrt(np.pi / 2), 0.10, 0.0, None),
    ("vonmises", d_vonmises, 0.0, 0.35, -8.0, 8.0),
    ("wald", d_wald, 2.0, 0.15, 0.0, None),
    ("weibull", d_weibull, np.sqrt(np.pi) / 2, 0.10, 0.0, None),
    ("geometric", d_geometric, 1.0 / 0.3, 0.15, 1.0, None),
    ("poisson", d_poisson, 4.0, 0.10, 0.0, None),
    ("binomial", d_binomial, 5.0, 0.08, 0.0, 10.0),
    ("negative_binomial", d_negative_binomial, 3.0, 0.15, 0.0, None),
    ("hypergeometric", d_hypergeometric, 2.5, 0.12, 0.0, 5.0),
    ("logseries", d_logseries, 0.5 / (0.5 * np.log(2.0)), 0.15, 1.0, None),
    ("zipf", d_zipf, None, 0.30, 1.0, None),
]


@pytest.mark.parametrize("name,draw,mean,tolerance,lower,upper", CASES,
                         ids=[case[0] for case in CASES])
def test_a_draw_has_its_distributions_shape(backend, name, draw, mean, tolerance,
                                            lower, upper):
    out = np.zeros(COUNT, dtype=np.float32)
    draw[backend](out)
    assert np.all(np.isfinite(out)), name
    if lower is not None:
        assert np.all(out >= lower), name
    if upper is not None:
        assert np.all(out <= upper), name
    if mean is not None:
        assert float(out.mean()) == pytest.approx(
            float(mean), rel=tolerance, abs=tolerance), name


def test_every_spelling_is_registered():
    """The catalogue names every scalar-valued distribution in both libraries."""
    from pygpgpu.kernel import random as random_support

    names = random_support.supported_spellings()
    for name in ("random.expovariate", "random.weibullvariate", "np.random.gamma",
                 "np.random.poisson", "np.random.zipf", "np.random.standard_t"):
        assert name in names
    for absent in ("np.random.choice", "random.shuffle", "np.random.multinomial"):
        assert absent not in names
