# /// script
# requires-python = ">=3.14"
# dependencies = ["marimo>=0.24.0", "sympy", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import morimo as  mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ridge Regression — Closed-Form Solution

    **References**

    - [Mathematics for Machine Learning](https://mml-book.github.io/) — Ch. 9.2
    - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Ch. 3.1.4
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.  Objective Function

    For design matrix $X \in \mathbb{R}^{n \times d}$, targets $y \in \mathbb{R}^{n}$,
    weights $w \in \mathbb{R}^{d}$, and regularization $\lambda > 0$:

    \[
    J(w) = (Xw - y)^T (Xw - y) + \lambda w^T w
    \]

    The first term is the **data-fit** (sum of squared errors).
    The second term is the **$\ell_2$ penalty** that shrinks weights toward zero.
    """)
    return


@app.cell
def _():
    import sympy as sp

    n, d = sp.symbols("n d", integer=True, positive=True)
    lam = sp.symbols("lambda", positive=True, real=True)

    X = sp.MatrixSymbol("X", n, d)
    y_vec = sp.MatrixSymbol("y", n, 1)
    w_vec = sp.MatrixSymbol("w", d, 1)

    # Objective: (Xw - y)^T (Xw - y) + lambda * w^T w
    residual = X * w_vec - y_vec
    data_term = residual.T * residual
    penalty_term = lam * (w_vec.T * w_vec)

    J = data_term + penalty_term
    J[0, 0]
    return d, lam, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.  Stationarity Condition

    Set the gradient to zero:

    \[
    \nabla_w J = 2 X^T (X w - y) + 2 \lambda w = 0
    \]

    Expanding:

    \[
    X^T X w - X^T y + \lambda w = 0
    \]

    \[
    (X^T X + \lambda I) w = X^T y
    \]
    """)
    return


@app.cell
def _(d, lam, sp):
    I_d = sp.Identity(d)

    lhs = sp.MatAdd(sp.MatMul(sp.Transpose(sp.MatrixSymbol("X", sp.Symbol("n"), d)), sp.MatrixSymbol("X", sp.Symbol("n"), d)), sp.MatMul(lam, I_d))
    rhs = sp.MatMul(sp.Transpose(sp.MatrixSymbol("X", sp.Symbol("n"), d)), sp.MatrixSymbol("y", sp.Symbol("n"), 1))

    sp.Eq(lhs * sp.MatrixSymbol("w", d, 1), rhs, evaluate=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  Closed-Form Solution

    \[
    w^* = (X^T X + \lambda I)^{-1} X^T y
    \]

    This is the **ridge-regression** (or Tikhonov-regularized) solution.

    - When $\lambda = 0$, it reduces to the ordinary least-squares solution $w^* = (X^T X)^{-1} X^T y$.
    - When $\lambda > 0$, the matrix $X^T X + \lambda I$ is always invertible,
      even when $X^T X$ is singular (e.g., $d > n$).
    """)
    return


@app.cell
def _(d, lam, sp):
    X_sym = sp.MatrixSymbol("X", sp.Symbol("n"), d)
    y_sym = sp.MatrixSymbol("y", sp.Symbol("n"), 1)
    I_sym = sp.Identity(d)

    ridge_solution = (X_sym.T * X_sym + lam * I_sym).inv() * X_sym.T * y_sym

    sp.Eq(sp.MatrixSymbol("w", d, 1), ridge_solution, evaluate=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Numerical Verification

    Verify the closed-form solution on a small dataset:
    """)
    return


@app.cell
def _():
    import numpy as np

    np.random.seed(42)
    n_data, d_data = 50, 5
    X_data = np.random.randn(n_data, d_data)
    w_true = np.array([1.5, -0.8, 0.0, 2.0, -1.2])
    y_data = X_data @ w_true + 0.3 * np.random.randn(n_data)

    lambda_val = 0.5

    # Closed-form ridge solution
    I_mat = np.eye(d_data)
    w_ridge = np.linalg.solve(X_data.T @ X_data + lambda_val * I_mat, X_data.T @ y_data)

    # OLS for comparison
    w_ols = np.linalg.solve(X_data.T @ X_data, X_data.T @ y_data)

    w_ridge, w_true, w_ols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  Effect of $\lambda$ on the Solution Norm

    As $\lambda$ increases, $\|w\|_2$ shrinks (regularization strength):
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_data, d_data = 50, 5
    X_data = np.random.randn(n_data, d_data)
    w_true = np.array([1.5, -0.8, 0.0, 2.0, -1.2])
    y_data = X_data @ w_true + 0.3 * np.random.randn(n_data)

    lambdas = np.logspace(-3, 3, 50)
    norms = []
    I_mat = np.eye(d_data)

    for lam_v in lambdas:
        w_est = np.linalg.solve(X_data.T @ X_data + lam_v * I_mat, X_data.T @ y_data)
        norms.append(np.linalg.norm(w_est))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(lambdas, norms, "b-", linewidth=2)
    ax.axhline(np.linalg.norm(w_true), color="gray", linestyle="--",
               label=f"$\\|w_{{\\text{{true}}}}\\| = {np.linalg.norm(w_true):.2f}$")
    ax.set_xlabel("$\\lambda$ (regularization strength)", fontsize=12)
    ax.set_ylabel("$\\|\\hat{{w}}\\|_2$", fontsize=12)
    ax.set_title("Effect of $\\lambda$ on Solution Norm", fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
