# /// script
# requires-python = ">=3.14"
# dependencies = ["marimo>=0.24.0", "sympy", "numpy", "matplotlib", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Convolution — Symbolic Derivation with SymPy

    **References**

    - [SymPy Convolution](https://docs.sympy.org/latest/modules/integrals/integrals.html)
    - [Convolution in Linear Systems](https://dynamics-and-control.readthedocs.io/en/latest/1_Dynamics/3_Linear_systems/Convolution.html)
    - [Implementing Convolution using SymPy](https://codereview.stackexchange.com/questions/174538/implementing-convolution-using-sympy)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.  Definition

    The **convolution** of two functions $f$ and $g$ is:

    \[
    (f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau
    \]

    For **causal** functions (zero for $t < 0$), the limits simplify to:

    \[
    (f * g)(t) = \int_{0}^{t} f(\tau) \, g(t - \tau) \, d\tau
    \]

    **Key properties:**
    - Commutative: $f * g = g * f$
    - Associative: $f * (g * h) = (f * g) * h$
    - Distributive: $f * (g + h) = f * g + f * h$
    """)
    return


@app.cell
def _():
    import sympy as sp

    t = sp.symbols("t", real=True)
    tau = sp.symbols("tau", real=True)

    t, tau
    return sp, t, tau


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.  Example 1 — Convolution of Two Exponentials

    $f(t) = e^{-t}$, $g(t) = e^{-2t}$, for $t \geq 0$:

    \[
    (f * g)(t) = \int_{0}^{t} e^{-\tau} \, e^{-2(t - \tau)} \, d\tau
    \]
    """)
    return


@app.cell
def _(sp, t, tau):
    f_exp = sp.exp(-tau)
    g_exp = sp.exp(-2 * (t - tau))

    integrand = f_exp * g_exp
    conv_exp = sp.integrate(integrand, (tau, 0, t))

    (
        sp.Eq(sp.Symbol("f(tau)"), f_exp),
        sp.Eq(sp.Symbol("g(t-tau)"), g_exp),
        sp.Eq(sp.Symbol("(f * g)(t)"), sp.simplify(conv_exp)),
    )
    return conv_exp, f_exp, g_exp, integrand


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  Example 2 — Convolution of Gaussian with Itself

    The convolution of two Gaussians is a Gaussian:

    \[
    \mathcal{N}(0, \sigma_1^2) * \mathcal{N}(0, \sigma_2^2) = \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)
    \]

    This is a key property underlying Gaussian processes and diffusion.
    """)
    return


@app.cell
def _(sp, t, tau):
    sig1 = sp.symbols("sigma_1", positive=True, real=True)
    sig2 = sp.symbols("sigma_2", positive=True, real=True)

    f_gauss = sp.exp(-(tau ** 2) / (2 * sig1 ** 2)) / (sp.sqrt(2 * sp.pi) * sig1)
    g_gauss = sp.exp(-((t - tau) ** 2) / (2 * sig2 ** 2)) / (sp.sqrt(2 * sp.pi) * sig2)

    # Compute convolution via integral (symbolic; may be slow for general case)
    integrand_gauss = sp.simplify(f_gauss * g_gauss)

    (
        sp.Eq(sp.Symbol("f(tau)"), f_gauss),
        sp.Eq(sp.Symbol("g(t-tau)"), g_gauss),
    )
    return f_gauss, g_gauss, integrand_gauss, sig1, sig2


@app.cell
def _(f_gauss, g_gauss, sig1, sig2, sp, t, tau):
    # Numerically verify the variance-addition property
    conv_gauss_sym = sp.integrate(
        f_gauss * g_gauss,
        (tau, -sp.oo, sp.oo),
    )

    # Expected result: N(0, sigma_1^2 + sigma_2^2)
    expected = sp.exp(-(t ** 2) / (2 * (sig1 ** 2 + sig2 ** 2))) / (
        sp.sqrt(2 * sp.pi * (sig1 ** 2 + sig2 ** 2))
    )

    check = sp.simplify(conv_gauss_sym - expected)

    (
        sp.Eq(sp.Symbol("(f * g)(t)"), sp.simplify(conv_gauss_sym)),
        sp.Eq(sp.Symbol("Expected"), expected),
        f"Difference simplifies to zero: {check == 0}",
    )
    return check, conv_gauss_sym, expected


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Example 3 — Convolution of Uniform with Exponential

    $f(t) = \mathbb{1}_{[0,1]}(t)$ (unit pulse), $g(t) = e^{-t}$ for $t \geq 0$:

    The result is a **smoothed exponential** — a classic example of how
    convolution smooths signals.
    """)
    return


@app.cell
def _(sp, t, tau):
    # Uniform on [0,1] convolved with exponential
    f_uniform = sp.Piecewise((1, (tau >= 0) & (tau <= 1)), (0, True))
    g_exp2 = sp.exp(-(t - tau))

    conv_uniform_exp = sp.integrate(
        f_uniform * g_exp2,
        (tau, 0, t),
    )

    sp.simplify(conv_uniform_exp)
    return conv_uniform_exp, f_uniform, g_exp2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  Numerical Visualization

    Visualizing convolution as a sliding dot-product:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    def f_signal(t):
        return np.where((t >= 0) & (t <= 1), 1.0, 0.0)

    def g_signal(t):
        return np.where(t >= 0, np.exp(-t), 0.0)

    t_vals = np.linspace(-0.5, 5, 500)
    dt = t_vals[1] - t_vals[0]

    f_vals = f_signal(t_vals)
    g_vals = g_signal(t_vals)

    # Numerical convolution
    conv_vals = np.convolve(f_vals, g_vals, mode="full") * dt
    t_conv = np.linspace(-0.5 * 2, 5 * 2, len(conv_vals))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    axes[0].plot(t_vals, f_vals, "b-", linewidth=2)
    axes[0].fill_between(t_vals, 0, f_vals, alpha=0.2, color="blue")
    axes[0].set_title("$f(t)$: Unit pulse", fontsize=12)
    axes[0].set_xlabel("$t$")
    axes[0].set_ylim(0, 1.2)

    axes[1].plot(t_vals, g_vals, "r-", linewidth=2)
    axes[1].fill_between(t_vals, 0, g_vals, alpha=0.2, color="red")
    axes[1].set_title("$g(t)$: Exponential decay", fontsize=12)
    axes[1].set_xlabel("$t$")

    axes[2].plot(t_vals, f_vals, "b-", linewidth=1.5, alpha=0.5, label="$f$")
    axes[2].plot(t_vals, g_vals[::-1], "r--", linewidth=1.5, alpha=0.5,
                 label="$g(-\\tau)$ flipped")
    axes[2].set_title("Sliding $g(t-\\tau)$", fontsize=12)
    axes[2].set_xlabel("$\\tau$")
    axes[2].legend(fontsize=9)

    axes[3].plot(t_conv, conv_vals, "purple", linewidth=2)
    axes[3].fill_between(t_conv, 0, conv_vals, alpha=0.15, color="purple")
    axes[3].set_title("$(f * g)(t)$: Result", fontsize=12)
    axes[3].set_xlabel("$t$")
    axes[3].set_xlim(-0.5, 5)

    plt.tight_layout()
    fig
    return (fig,)


if __name__ == "__main__":
    app.run()