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
    # Error Functions — Symbolic Derivation

    **References**

    - [SymPy Calculus](https://docs.sympy.org/latest/modules/integrals/integrals.html)
    - [Calculus in Python](https://www.askpython.com/python/examples/calculus-in-python)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.  Error Functions in ML

    Common error functions and their properties:

    | Function | Formula | Use Case |
    |----------|---------|----------|
    | MSE | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Regression |
    | MAE | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Robust regression |
    | Huber | $\begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta(|e| - \frac{1}{2}\delta) & |e| > \delta \end{cases}$ | Robust to outliers |
    | Cross-Entropy | $-\sum y_i \log(\hat{y}_i)$ | Classification |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.  MSE — Mean Squared Error

    For predictions $\hat{y}_i$ and targets $y_i$:

    \[
    E_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
    \]

    **Gradient** w.r.t. a single prediction $\hat{y}_k$:

    \[
    \frac{\partial E_{\text{MSE}}}{\partial \hat{y}_k} = \frac{2}{n}(\hat{y}_k - y_k)
    \]
    """)
    return


@app.cell
def _():
    import sympy as sp

    n = sp.Symbol("n", integer=True, positive=True)
    i, k = sp.symbols("i k", integer=True)
    y_hat = sp.IndexedBase("\\hat{y}", shape=(n,))
    y_true = sp.IndexedBase("y", shape=(n,))

    mse = (1 / n) * sp.Sum((y_hat[i] - y_true[i]) ** 2, (i, 1, n))

    # Gradient w.r.t. a single prediction
    y_hat_k = sp.Symbol("\\hat{y}_k", real=True)
    mse_single = (1 / n) * (y_hat_k - sp.Symbol("y_k", real=True)) ** 2
    grad_mse = sp.simplify(sp.diff(mse_single, y_hat_k))

    (
        sp.Eq(sp.Symbol("E_MSE"), mse),
        sp.Eq(sp.Symbol("frac{partial E_MSE}{partial hat{y}_k}"), grad_mse),
    )
    return grad_mse, i, k, mse, mse_single, n, sp, y_hat, y_hat_k, y_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  MAE — Mean Absolute Error

    \[
    E_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|
    \]

    The gradient is **piecewise constant** (subgradient at $e = 0$):

    \[
    \frac{\partial E_{\text{MAE}}}{\partial \hat{y}_k}
    = \frac{1}{n} \cdot \operatorname{sign}(\hat{y}_k - y_k)
    \]
    """)
    return


@app.cell
def _(sp):
    e = sp.Symbol("e", real=True)

    mae_single = sp.Abs(e)
    grad_mae = sp.diff(mae_single, e)

    (
        sp.Eq(sp.Symbol("E_MAE(single)"), mae_single),
        sp.Eq(sp.Symbol("frac{partial |e|}{partial e}"), grad_mae),
    )
    return e, grad_mae, mae_single


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Huber Loss — Smooth and Robust

    The Huber loss interpolates between MSE (small errors) and MAE (large errors):

    \[
    L_\delta(e) = \begin{cases}
    \frac{1}{2}e^2 & \text{if } |e| \leq \delta \\[4pt]
    \delta\left(|e| - \frac{1}{2}\delta\right) & \text{if } |e| > \delta
    \end{cases}
    \]

    Its gradient is **clipped**:

    \[
    \frac{\partial L_\delta}{\partial e} = \begin{cases}
    e & \text{if } |e| \leq \delta \\
    \delta \cdot \operatorname{sign}(e) & \text{if } |e| > \delta
    \end{cases}
    \]
    """)
    return


@app.cell
def _(sp):
    e = sp.Symbol("e", real=True)
    delta = sp.Symbol("delta", positive=True)

    huber = sp.Piecewise(
        (sp.Rational(1, 2) * e ** 2, sp.Abs(e) <= delta),
        (delta * (sp.Abs(e) - sp.Rational(1, 2) * delta), True),
    )

    grad_huber = sp.diff(huber, e)

    (
        sp.Eq(sp.Symbol("L_delta(e)"), huber),
        sp.Eq(sp.Symbol("frac{partial L_delta}{partial e}"), grad_huber),
    )
    return delta, e, grad_huber, huber


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  Visual Comparison

    MSE penalizes large errors quadratically, MAE linearly, and Huber
    smoothly transitions between them:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    e_vals = np.linspace(-4, 4, 500)
    delta_val = 1.5

    mse_vals = 0.5 * e_vals ** 2
    mae_vals = np.abs(e_vals)
    huber_vals = np.where(
        np.abs(e_vals) <= delta_val,
        0.5 * e_vals ** 2,
        delta_val * (np.abs(e_vals) - 0.5 * delta_val),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(e_vals, mse_vals, "b-", linewidth=2, label="MSE: $\\frac{1}{2}e^2$")
    ax1.plot(e_vals, mae_vals, "r-", linewidth=2, label="MAE: $|e|$")
    ax1.plot(e_vals, huber_vals, "g--", linewidth=2.5,
             label=f"Huber ($\\delta={delta_val}$)")
    ax1.set_xlabel("Error $e = \\hat{y} - y$", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Functions", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 8)

    ax2.plot(e_vals, e_vals, "b-", linewidth=2, label="MSE gradient: $e$")
    ax2.plot(e_vals, np.sign(e_vals), "r-", linewidth=2, label="MAE gradient: $\\operatorname{sign}(e)$")
    huber_grad = np.where(np.abs(e_vals) <= delta_val, e_vals, delta_val * np.sign(e_vals))
    ax2.plot(e_vals, huber_grad, "g--", linewidth=2.5,
             label=f"Huber grad ($\\delta={delta_val}$)")
    ax2.set_xlabel("Error $e$", fontsize=12)
    ax2.set_ylabel("Gradient", fontsize=12)
    ax2.set_title("Gradients", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-3, 3)

    plt.tight_layout()
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.  Summary: Which Error Function to Use?

    | Criterion | Choice |
    |-----------|--------|
    | Gaussian noise assumption | **MSE** |
    | Outliers present | **MAE** or **Huber** |
    | Binary classification | **Binary Cross-Entropy** |
    | Multi-class classification | **Categorical Cross-Entropy** |
    | Need smooth gradients everywhere | **Huber** ($\delta \approx 1.0$) |
    """)
    return


if __name__ == "__main__":
    app.run()