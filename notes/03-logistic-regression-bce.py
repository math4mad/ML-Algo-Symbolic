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
    # Logistic Regression — Binary Cross-Entropy

    **References**

    - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Ch. 4.3
    - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) — Ch. 10
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.  Model Definition

    **Logit (linear predictor):**

    \[
    z = w x + b
    \]

    **Sigmoid activation** (maps $z$ to probability $p \in (0, 1)$):

    \[
    p = \sigma(z) = \frac{1}{1 + e^{-z}}
    \]

    **Binary Cross-Entropy loss** for one example $(x, y)$ with $y \in \{0, 1\}$:

    \[
    \mathcal{L} = -\bigl[ y \log(p) + (1 - y) \log(1 - p) \bigr]
    \]
    """)
    return


@app.cell
def _():
    import sympy as sp

    x, y, w, b = sp.symbols("x y w b", real=True)

    z = w * x + b
    p = 1 / (1 + sp.exp(-z))

    binary_cross_entropy = -(y * sp.log(p) + (1 - y) * sp.log(1 - p))

    (
        sp.Eq(sp.Symbol("z"), z),
        sp.Eq(sp.Symbol("p"), p),
        sp.Eq(sp.Symbol("mathcal{L}"), binary_cross_entropy),
    )
    return b, binary_cross_entropy, p, sp, w, x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.  Gradient w.r.t. $z$ (the "surprise" gradient)

    This is the key simplification that makes logistic regression elegant:

    \[
    \frac{\partial \mathcal{L}}{\partial z} = p - y
    \]
    """)
    return


@app.cell
def _(binary_cross_entropy, sp, z):
    grad_z = sp.simplify(sp.diff(binary_cross_entropy, z))

    sp.Eq(sp.Symbol("frac{partial L}{partial z}"), grad_z)
    return grad_z,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.  Gradients w.r.t. Parameters $w$ and $b$

    By the chain rule:

    \[
    \frac{\partial \mathcal{L}}{\partial w}
    = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w}
    = (p - y) \cdot x
    \]

    \[
    \frac{\partial \mathcal{L}}{\partial b}
    = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial b}
    = p - y
    \]
    """)
    return


@app.cell
def _(b, binary_cross_entropy, sp, w):
    grad_w = sp.simplify(sp.diff(binary_cross_entropy, w))
    grad_b = sp.simplify(sp.diff(binary_cross_entropy, b))

    (
        sp.Eq(sp.Symbol("frac{partial L}{partial w}"), grad_w),
        sp.Eq(sp.Symbol("frac{partial L}{partial b}"), grad_b),
    )
    return grad_b, grad_w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.  Verification of Equivalent Forms

    Confirm $\frac{\partial \mathcal{L}}{\partial w} = x(p - y)$ and
    $\frac{\partial \mathcal{L}}{\partial b} = p - y$:
    """)
    return


@app.cell
def _(p, sp, w, x, y):
    grad_w_form1 = sp.simplify(sp.diff(-(y * sp.log(p) + (1 - y) * sp.log(1 - p)), w))
    grad_w_form2 = x * (p - y)

    grad_b_form1 = sp.simplify(sp.diff(-(y * sp.log(p) + (1 - y) * sp.log(1 - p)), b))
    grad_b_form2 = p - y

    check_w = sp.simplify(grad_w_form1 - grad_w_form2) == 0
    check_b = sp.simplify(grad_b_form1 - grad_b_form2) == 0

    f"∂L/∂w equivalent: {check_w}", f"∂L/∂b equivalent: {check_b}"
    return check_b, check_w, grad_b_form1, grad_b_form2, grad_w_form1, grad_w_form2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.  Gradient Descent Update Rules

    \[
    \begin{aligned}
    w &\leftarrow w - \eta \cdot x(p - y) \\[4pt]
    b &\leftarrow b - \eta \cdot (p - y)
    \end{aligned}
    \]

    where $p = \sigma(wx + b) = \frac{1}{1 + e^{-(wx+b)}}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.  Visualizing the Sigmoid and Loss

    The sigmoid function and cross-entropy loss as a function of $z$:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z_vals = np.linspace(-6, 6, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: sigmoid
    ax1.plot(z_vals, sigmoid(z_vals), "b-", linewidth=2.5)
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax1.fill_between(z_vals, 0, sigmoid(z_vals), alpha=0.15, color="blue")
    ax1.set_xlabel("$z = wx + b$", fontsize=12)
    ax1.set_ylabel("$\\sigma(z)$", fontsize=12)
    ax1.set_title("Sigmoid $\\sigma(z)$", fontsize=13)

    # Right: cross-entropy loss for y=0 and y=1
    p_vals = sigmoid(z_vals)
    loss_y1 = -np.log(p_vals + 1e-15)
    loss_y0 = -np.log(1 - p_vals + 1e-15)

    ax2.plot(z_vals, loss_y1, "b-", linewidth=2, label="$y=1$: $-\\log(p)$")
    ax2.plot(z_vals, loss_y0, "r-", linewidth=2, label="$y=0$: $-\\log(1-p)$")
    ax2.set_xlabel("$z = wx + b$", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Binary Cross-Entropy Loss", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 6)

    plt.tight_layout()
    fig
    return (fig,)


if __name__ == "__main__":
    app.run()