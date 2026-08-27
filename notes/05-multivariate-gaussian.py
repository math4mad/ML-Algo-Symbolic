# /// script
# requires-python = ">=3.14"
# dependencies = ["marimo>=0.24.0", "sympy", "numpy", "matplotlib", "scipy"]
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
    # Multivariate Gaussian — Joint, Marginal, Conditional

    **References**

    - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Ch. 2.3
    - [Mathematics for Machine Learning](https://mml-book.github.io/) — Ch. 6.5
    - [Derive the distribution of two Gaussian variables' ratio](https://niwaka-ame.github.io/articles/sympy-gaussian-quotient.html)
    """)
    return


@app.cell
def _():
    import sympy as sp
    from sympy.stats import (
        MultivariateNormal,
        Normal,
        density,
        marginal_distribution,
        Expectation,
        covariance,
    )

    mu_x, mu_y = sp.symbols("mu_x mu_y", real=True)
    sigma_x = sp.symbols("sigma_x", positive=True, real=True)
    sigma_y = sp.symbols("sigma_y", positive=True, real=True)
    rho = sp.symbols("rho", real=True)

    x, y = sp.symbols("x y", real=True)

    # Covariance matrix
    Sigma = sp.Matrix([
        [sigma_x, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y],
    ])

    mu = sp.Matrix([mu_x, mu_y])

    MV = MultivariateNormal("MV", mu, Sigma)

    (
        sp.Eq(sp.Symbol("mu"), mu),
        sp.Eq(sp.Symbol("Sigma"), Sigma),
    )
    return (
        MV,
        MultivariateNormal,
        density,
        marginal_distribution,
        mu_x,
        mu_y,
        rho,
        sigma_x,
        sigma_y,
        sp,
        x,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.  Joint Density

    \[
    p(x, y) = \frac{1}{2\pi\sqrt{|\Sigma|}}
    \exp\!\left(-\frac{1}{2}
    \begin{bmatrix} x - \mu_x & y - \mu_y \end{bmatrix}
    \Sigma^{-1}
    \begin{bmatrix} x - \mu_x \\ y - \mu_y \end{bmatrix}
    \right)
    \]
    """)
    return


@app.cell
def _(MV, density, sp, x, y):
    joint_pdf = density(MV)(x, y)

    sp.simplify(joint_pdf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.  Marginal Distributions

    The marginals of a multivariate Gaussian are Gaussian:

    \[
    p(x) = \mathcal{N}(x \mid \mu_x, \sigma_x),\qquad
    p(y) = \mathcal{N}(y \mid \mu_y, \sigma_y)
    \]
    """)
    return


@app.cell
def _(MV, marginal_distribution, sp):
    marg_x = sp.simplify(marginal_distribution(MV, MV[0])(sp.Symbol("x")))
    marg_y = sp.simplify(marginal_distribution(MV, MV[1])(sp.Symbol("y")))

    (marg_x, marg_y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  Conditional Distribution

    For a bivariate Gaussian, the conditional $p(x \mid y = c)$ is:

    \[
    x \mid y = c \;\sim\; \mathcal{N}\!\left(
    \mu_x + \rho\frac{\sigma_x}{\sigma_y}(c - \mu_y),\;
    \sigma_x^2(1 - \rho^2)
    \right)
    \]
    """)
    return


@app.cell
def _(mu_x, mu_y, rho, sigma_x, sigma_y, sp):
    c = sp.Symbol("c", real=True)

    cond_mean = mu_x + rho * sigma_x / sigma_y * (c - mu_y)
    cond_var = sigma_x ** 2 * (1 - rho ** 2)

    (
        sp.Eq(sp.Symbol("E[X | Y=c]"), sp.simplify(cond_mean)),
        sp.Eq(sp.Symbol("Var[X | Y=c]"), sp.simplify(cond_var)),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Interactive Visualization

    Adjust $y$ to see how the conditional distribution changes:
    """)
    return


@app.cell
def _(mo):
    y_slider = mo.ui.slider(
        start=-3.0, stop=3.0, step=0.05, value=0.0,
        label="Fixed value of $y$",
    )
    y_slider
    return (y_slider,)


@app.cell
def _(y_slider):
    import numpy as np
    import matplotlib.pyplot as plt

    mu_x_val, mu_y_val = 0.0, 0.0
    sigma_x_val, sigma_y_val = 1.0, 1.4
    rho_val = 0.7

    fixed_y = y_slider.value

    cond_mean_val = mu_x_val + rho_val * sigma_x_val / sigma_y_val * (fixed_y - mu_y_val)
    cond_var_val = sigma_x_val ** 2 * (1 - rho_val ** 2)
    cond_std_val = np.sqrt(cond_var_val)

    x_vals = np.linspace(-4, 4, 250)
    y_vals = np.linspace(-4, 4, 250)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)

    cov_mat = np.array([
        [sigma_x_val ** 2, rho_val * sigma_x_val * sigma_y_val],
        [rho_val * sigma_x_val * sigma_y_val, sigma_y_val ** 2],
    ])
    inv_cov = np.linalg.inv(cov_mat)
    centered = np.stack([grid_x - mu_x_val, grid_y - mu_y_val], axis=-1)
    joint_density = (
        np.exp(-0.5 * np.einsum("...i,ij,...j->...", centered, inv_cov, centered))
        / (2 * np.pi * np.sqrt(np.linalg.det(cov_mat)))
    )

    cond_density = (
        np.exp(-0.5 * ((x_vals - cond_mean_val) / cond_std_val) ** 2)
        / (np.sqrt(2 * np.pi) * cond_std_val)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    c = ax1.contourf(grid_x, grid_y, joint_density, levels=20, cmap="viridis")
    ax1.axhline(fixed_y, color="white", linewidth=2.5,
                label=f"$y = {fixed_y:.2f}$")
    ax1.set(xlabel="$x$", ylabel="$y$", title="Joint density $p(x, y)$")
    ax1.legend(fontsize=11)

    ax2.plot(x_vals, cond_density, color="#e45756", linewidth=3)
    ax2.axvline(cond_mean_val, color="#2f4858", linestyle="--", linewidth=1.5,
                label=f"$\\mu_{{X|Y}} = {cond_mean_val:.2f}$")
    ax2.fill_between(x_vals, 0, cond_density, alpha=0.15, color="#e45756")
    ax2.set(xlabel="$x$", ylabel="Density",
            title=f"$p(x \\mid y = {fixed_y:.2f})$")
    ax2.legend(fontsize=11)

    fig.colorbar(c, ax=ax1, label="Density")
    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  General $n$-Dimensional Case

    For $X \sim \mathcal{N}(\mu, \Sigma)$ with $\Sigma$ full-rank:

    \[
    p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}}
    \exp\!\left(-\frac{1}{2}(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)
    \]

    **Partitioned form:** If we partition $X = \begin{bmatrix} X_a \\ X_b \end{bmatrix}$:

    \[
    \begin{aligned}
    \mu &= \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}, &
    \Sigma &= \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix}
    \end{aligned}
    \]

    Then:

    \[
    X_a \mid X_b = x_b \;\sim\; \mathcal{N}(
    \mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b),\;
    \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}
    )
    \]
    """)
    return


@app.cell
def _(MultivariateNormal, density, sp):
    n_dim = sp.Symbol("n", integer=True, positive=True)
    mu_vec = sp.MatrixSymbol("mu", n_dim, 1)
    Sigma_mat = sp.MatrixSymbol("Sigma", n_dim, n_dim)
    obs = sp.MatrixSymbol("x", n_dim, 1)

    X_mv = MultivariateNormal("X", mu_vec, Sigma_mat)
    pdf_nd = density(X_mv)(obs)

    pdf_nd
    return


if __name__ == "__main__":
    app.run()
