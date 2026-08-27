# /// script
# requires-python = ">=3.14"
# dependencies = ["marimo>=0.24.0", "sympy", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian Distribution — MLE Derivation

    **References**

    - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Ch. 2.3
    - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) — Ch. 4
    """)
    return


@app.cell
def _():
    import sympy as sp
    from sympy.stats import Normal, density, Expectation, variance, std

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

    Given i.i.d. observations $\{x_1, \ldots, x_n\}$, the likelihood is:

    \[
    \mathcal{L}(\mu, \sigma) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}}
    \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
    \]

    The **log-likelihood** is:

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
    ## 2.  MLE for $\mu$ — Score Equation

    \[
    \frac{\partial \ell}{\partial \mu}
    = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0
    \]

    Solving:

    \[
    \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i
    \]
    """)
    return


@app.cell
def _(i, log_likelihood, mu_sym, n, sp, x_i):
    dmu = sp.simplify(sp.diff(log_likelihood.function, mu_sym))

    # Solve: dmu = 0 => sum(x_i - mu) = 0 => mu = sum(x_i) / n
    mu_mle = sp.solve(
        sp.Eq(sp.Sum(x_i[i] - mu_sym, (i, 1, n)), 0),
        mu_sym,
    )

    (
        sp.Eq(sp.Symbol("frac{partial ell}{partial mu}"), sp.Sum(dmu, (i, 1, n))),
        #sp.Eq(sp.Symbol("hat{mu}_{MLE}"), mu_mle),
    )

    mu_mle
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  MLE for $\sigma^2$ — Score Equation

    \[
    \frac{\partial \ell}{\partial \sigma}
    = -\frac{n}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^{n}(x_i - \mu)^2 = 0
    \]

    Solving:

    \[
    \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2
    \]

    ⚠️ This is a **biased** estimator. The unbiased estimator uses $n-1$:
    $\hat{\sigma}^2_{\text{unbiased}} = \frac{1}{n-1}\sum(x_i - \bar{x})^2$.
    """)
    return


@app.cell
def _(i, log_likelihood, mu_sym, n, sigma_sym, sp, x_i):
    dsigma = sp.simplify(sp.diff(log_likelihood.function, sigma_sym))

    # Solve: -n/sigma + 1/sigma^3 * sum(x_i - mu)^2 = 0
    # => sigma^2 = (1/n) * sum(x_i - mu)^2
    sigma2_mle = sp.solve(
        sp.Eq(
            -n / sigma_sym + sp.Sum((x_i[i] - mu_sym) ** 2, (i, 1, n)) / sigma_sym ** 3,
            0,
        ),
        sigma_sym,
    )

    (
        sp.Eq(sp.Symbol("frac{partial ell}{partial sigma}"), sp.Sum(dsigma, (i, 1, n))),
        sp.Eq(sp.Symbol("hat{sigma}^2_{MLE}"), sigma2_mle[1] ** 2),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Numerical Verification

    Verify MLE estimates against known parameters:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, size=100)

    mu_hat = np.mean(data)
    sigma_hat = np.std(data)           # biased MLE
    sigma_hat_unbiased = np.std(data, ddof=1)  # unbiased

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=20, density=True, alpha=0.5, color="steelblue", label="Data histogram")

    x_plot = np.linspace(data.min() - 1, data.max() + 1, 300)
    ax.plot(x_plot,
            1 / (np.sqrt(2 * np.pi) * true_sigma) * np.exp(-0.5 * ((x_plot - true_mu) / true_sigma) ** 2),
            "k-", linewidth=2.5, label="True density")
    ax.plot(x_plot,
            1 / (np.sqrt(2 * np.pi) * sigma_hat) * np.exp(-0.5 * ((x_plot - mu_hat) / sigma_hat) ** 2),
            "r--", linewidth=2, label="MLE fit")

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Gaussian MLE Fit", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  Summary of Results

    | Parameter | MLE Estimator | Unbiased Estimator |
    |-----------|--------------|-------------------|
    | $\mu$ | $\bar{x} = \frac{1}{n}\sum x_i$ | $\bar{x}$ (same) |
    | $\sigma^2$ | $\frac{1}{n}\sum (x_i - \bar{x})^2$ | $\frac{1}{n-1}\sum (x_i - \bar{x})^2$ |
    """)
    return


if __name__ == "__main__":
    app.run()
