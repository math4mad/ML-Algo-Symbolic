# /// script
# requires-python = ">=3.14"
# dependencies = ["marimo>=0.24.0", "sympy", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian Log-Likelihood — Symbolic Derivation

    Given i.i.d. observations $\{x_1, \ldots, x_n\}$ from a
    $N(\mu, \sigma^2)$ distribution, we derive the log-likelihood
    symbolically, compute its gradients, solve for the MLE, and
    verify numerically.

    **References**

    - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Ch. 2.3
    - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) — Ch. 4
    """)
    return


@app.cell
def _():
    import sympy as sp
    from sympy.stats import Normal, density

    mu = sp.Symbol("mu", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    x = sp.Symbol("x", real=True)

    X = Normal("X", mu, sigma)

    (
        sp.Eq(sp.Symbol("X"), X),
        sp.Eq(sp.Symbol("p(x)"), density(X)(x)),
    )
    return (sp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.  Log-Likelihood for $n$ Observations

    Likelihood:

    \[
    \mathcal{L}(\mu, \sigma) = \prod_{i=1}^{n}
    \frac{1}{\sqrt{2\pi\sigma^2}}
    \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
    \]

    Log-likelihood:

    \[
    \ell(\mu, \sigma) = -\frac{n}{2}\log(2\pi) - n\log\sigma
    - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
    \]
    """)
    return


@app.cell
def _(sp):
    n = sp.Symbol("n", integer=True, positive=True)
    i = sp.symbols("i", integer=True)
    mu_sym = sp.Symbol("mu", real=True)
    sigma_sym = sp.Symbol("sigma", positive=True)
    x_i = sp.IndexedBase("x", shape=(n,))

    log_likelihood = sp.Sum(
        -sp.log(sigma_sym)
        - sp.log(2 * sp.pi) / 2
        - (x_i[i] - mu_sym) ** 2 / (2 * sigma_sym ** 2),
        (i, 1, n),
    )

    log_likelihood
    return i, log_likelihood, mu_sym, n, sigma_sym, x_i


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.  Gradient w.r.t. $\mu$ — Score Equation

    \[
    \frac{\partial \ell}{\partial \mu}
    = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0
    \quad\Longrightarrow\quad
    \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i
    \]
    """)
    return


@app.cell
def _(i, log_likelihood, mu_sym, n, sp, x_i):
    dmu_term = sp.simplify(sp.diff(log_likelihood.function, mu_sym))
    dmu_sum = sp.Sum(dmu_term, (i, 1, n))

    mu_mle = sp.solve(
        sp.Eq(sp.Sum(x_i[i] - mu_sym, (i, 1, n)), 0),
        mu_sym,
    )

    (
        sp.Eq(sp.Symbol("frac{partial ell}{partial mu}"), dmu_sum),
        sp.Eq(sp.Symbol("hat{mu}_{MLE}"), mu_mle[0]),
    )
    return dmu_sum, mu_mle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  Gradient w.r.t. $\sigma$ — Score Equation

    \[
    \frac{\partial \ell}{\partial \sigma}
    = -\frac{n}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^{n}(x_i - \mu)^2 = 0
    \quad\Longrightarrow\quad
    \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2
    \]

    ⚠️ The MLE of $\sigma^2$ is **biased**. The unbiased estimator uses $n-1$:
    $\hat{\sigma}^2_{\text{unbiased}} = \frac{1}{n-1}\sum(x_i - \bar{x})^2$.
    """)
    return


@app.cell
def _(i, log_likelihood, mu_sym, n, sigma_sym, sp, x_i):
    dsigma_term = sp.simplify(sp.diff(log_likelihood.function, sigma_sym))
    dsigma_sum = sp.Sum(dsigma_term, (i, 1, n))

    sigma_solutions = sp.solve(
        sp.Eq(
            -n / sigma_sym
            + sp.Sum((x_i[i] - mu_sym) ** 2, (i, 1, n)) / sigma_sym ** 3,
            0,
        ),
        sigma_sym,
    )
    sigma2_mle = sigma_solutions[1] ** 2

    (
        sp.Eq(sp.Symbol("frac{partial ell}{partial sigma}"), dsigma_sum),
        sp.Eq(sp.Symbol("hat{sigma}^2_{MLE}"), sigma2_mle),
    )
    return dsigma_sum, sigma2_mle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Numerical Verification

    Generate samples from $N(5, 2^2)$ and verify the MLE estimates
    recover the true parameters.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, size=200)

    mu_hat = np.mean(data)
    sigma_hat_biased = np.std(data)
    sigma_hat_unbiased = np.std(data, ddof=1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        data, bins=25, density=True, alpha=0.45,
        color="steelblue", edgecolor="white", label="Data histogram",
    )

    x_plot = np.linspace(data.min() - 1, data.max() + 1, 400)
    true_pdf = (
        1
        / (np.sqrt(2 * np.pi) * true_sigma)
        * np.exp(-0.5 * ((x_plot - true_mu) / true_sigma) ** 2)
    )
    mle_pdf = (
        1
        / (np.sqrt(2 * np.pi) * sigma_hat_biased)
        * np.exp(-0.5 * ((x_plot - mu_hat) / sigma_hat_biased) ** 2)
    )

    ax.plot(x_plot, true_pdf, "k-", linewidth=2.5, label="True density")
    ax.plot(x_plot, mle_pdf, "r--", linewidth=2, label="MLE fit")

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Gaussian MLE  —  "
        f"$\\hat\\mu$={mu_hat:.3f} (true {true_mu}),  "
        f"$\\hat\\sigma$={sigma_hat_biased:.3f} (true {true_sigma})",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  Summary

    | Parameter | MLE Estimator | Unbiased Estimator |
    |-----------|--------------|-------------------|
    | $\mu$ | $\bar{x} = \frac{1}{n}\sum x_i$ | $\bar{x}$ (same) |
    | $\sigma^2$ | $\frac{1}{n}\sum (x_i - \bar{x})^2$ | $\frac{1}{n-1}\sum (x_i - \bar{x})^2$ |
    """)
    return


if __name__ == "__main__":
    app.run()
