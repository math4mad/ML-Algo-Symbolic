# /// script
# requires-python = ">=3.14"
# dependencies = ["marimo>=0.24.0", "sympy", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Regression — Mean Squared Error

    **References**

    - [Mathematics for Machine Learning](https://mml-book.github.io/) — Ch. 9
    - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) — Ch. 11
    """)
    return


@app.cell
def _():
    import sympy as sp

    return (sp,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.  Model and Loss

    For a single training example $(x, y)$, the linear model predicts:

    \[
    \hat{y} = w x + b
    \]

    The squared-error loss (with $\frac{1}{2}$ for cleaner gradients):

    \[
    \mathcal{L} = \frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2}(w x + b - y)^2
    \]
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(sp):


    # Symbolic variables
    x, y, w, b = sp.symbols("x y w b", real=True)

    # Prediction and loss
    y_hat = w * x + b
    loss = sp.Rational(1, 2) * (y_hat - y) ** 2
    sp.Eq(sp.Symbol("hat{y}"), y_hat), sp.Eq(sp.Symbol("mathcal{L}"), loss)
    return b, loss, w, x, y, y_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.  Gradient Derivation

    Compute partial derivatives of $\mathcal{L}$ w.r.t. $w$ and $b$:
    """)
    return


@app.cell
def _(b, loss, sp, w):
    grad_w = sp.simplify(sp.diff(loss, w))
    grad_b = sp.simplify(sp.diff(loss, b))

    grad_w_simplified = sp.simplify(grad_w)
    grad_b_simplified = sp.simplify(grad_b)

    # display(
    #     sp.Eq(sp.Symbol("frac{partial L}{partial w}"), grad_w_simplified),
    #     sp.Eq(sp.Symbol("frac{partial L}{partial b}"), grad_b_simplified),
    # )
    (
        sp.Eq(sp.Symbol("frac{partial L}{partial w}"), grad_w_simplified),
        sp.Eq(sp.Symbol("frac{partial L}{partial b}"), grad_b_simplified),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  Verifying Equivalent Forms

    The gradient $\frac{\partial \mathcal{L}}{\partial w} = x(b + wx - y)$
    is equivalent to $x(\hat{y} - y)$:
    """)
    return


@app.cell
def _(b, sp, w, x, y, y_hat):
    form1 = x * (b + w * x - y)
    form2 = x * (y_hat - y)
    equiv = sp.simplify(form1 - form2) == 0

    sp.Eq(form1, form2, evaluate=False), f"Equivalent: {equiv}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Gradient Descent Update Rules

    \[
    \begin{aligned}
    w &\leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}
        = w - \eta \cdot x(\hat{y} - y) \\[4pt]
    b &\leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
        = b - \eta \cdot (\hat{y} - y)
    \end{aligned}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  Extension to $n$ Training Examples

    For a dataset $\{(x_i, y_i)\}_{i=1}^n$, the total loss is:

    \[
    \mathcal{L}_{\text{total}} = \frac{1}{2} \sum_{i=1}^{n} (w x_i + b - y_i)^2
    \]

    With gradients:

    \[
    \frac{\partial \mathcal{L}_{\text{total}}}{\partial w}
    = \sum_{i=1}^{n} x_i (w x_i + b - y_i),\qquad
    \frac{\partial \mathcal{L}_{\text{total}}}{\partial b}
    = \sum_{i=1}^{n} (w x_i + b - y_i)
    \]
    """)
    return


@app.cell
def _(sp):
    n = sp.Symbol("n", integer=True, positive=True)
    idx = sp.symbols("idx", integer=True)
    x_i = sp.IndexedBase("x", shape=(n,))
    y_i = sp.IndexedBase("y", shape=(n,))
    w_sym, b_sym = sp.symbols("w b", real=True)

    total_loss_expr = sp.Rational(1, 2) * sp.Sum(
        (w_sym * x_i[idx] + b_sym - y_i[idx]) ** 2, (idx, 1, n)
    )
    total_loss_expr
    return b_sym, total_loss_expr, w_sym


@app.cell
def _(b_sym, sp, total_loss_expr, w_sym):
    grad_w_total = sp.simplify(sp.diff(total_loss_expr, w_sym))
    grad_b_total = sp.simplify(sp.diff(total_loss_expr, b_sym))

    # display(
    #     sp.Eq(sp.Symbol("frac{partial L_total}{partial w}"), grad_w_total),
    #     sp.Eq(sp.Symbol("frac{partial L_total}{partial b}"), grad_b_total),
    # )

    (
        sp.Eq(sp.Symbol("frac{partial L_total}{partial w}"), grad_w_total),
        sp.Eq(sp.Symbol("frac{partial L_total}{partial b}"), grad_b_total),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.  Visualizing the Loss Surface

    For a fixed dataset, plot $\mathcal{L}(w, b)$:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Toy data: y = 3x + 2 + noise
    np.random.seed(42)
    x_data = np.linspace(-2, 2, 20)
    y_true = 3 * x_data + 2 + np.random.randn(20) * 0.5

    W_grid, B_grid = np.meshgrid(np.linspace(-1, 5, 100), np.linspace(-2, 6, 100))
    Loss_grid = np.zeros_like(W_grid)
    for i in range(W_grid.shape[0]):
        for j in range(W_grid.shape[1]):
            pred = W_grid[i, j] * x_data + B_grid[i, j]
            Loss_grid[i, j] = 0.5 * np.mean((pred - y_true) ** 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(W_grid, B_grid, Loss_grid, levels=30, cmap="viridis")
    ax.scatter([3], [2], color="red", marker="*", s=200, edgecolors="white",
               label="True params $(w=3, b=2)$", zorder=5)
    ax.set_xlabel("$w$", fontsize=13)
    ax.set_ylabel("$b$", fontsize=13)
    ax.set_title("MSE Loss Surface $\\mathcal{L}(w, b)$", fontsize=14)
    fig.colorbar(c, ax=ax, label="Loss")
    ax.legend()
    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
