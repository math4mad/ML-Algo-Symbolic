# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sympy  as sy

    return (sy,)


@app.cell
def _():
    from sympy.abc import i, k, m
    from sympy import Sum, factorial, oo, IndexedBase, Symbol
    Sum(k, (k, 1, m))
    return (i,)


@app.cell
def _():
    # Experimental function definition retained for reference.
    # y = sy.Function('y')
    return


@app.cell
def _():
    # t = sy.IndexedBase('t')
    # w = sy.IndexedBase('w')
    # t_hat = sy.IndexedBase('\\hat{t}')
    # error = sy.Sum(w[i] * (t[i] - t_hat[i])**2, (i, 1, n))
    #error_latex = sy.latex(error)
    #error_latex
    #error

    return


@app.cell
def _(sy):
    n = sy.Symbol('n', integer=True, positive=True)
    t = sy.IndexedBase('t', shape=(n,))
    w = sy.IndexedBase('w', shape=(n,))
    x = sy.IndexedBase('x', shape=(n,))

    (t.shape, w.shape, x.shape)
    return n, t, w, x


@app.cell
def _():
    from sympy import MatrixSymbol, Matrix

    return


@app.cell
def _():
    # f_function = sy.Function('f')
    # f_xi_w = f_function(x[i], w)
    # f_xi_w_latex = sy.latex(f_xi_w)

    # f_xi_w_latex
    return


@app.cell
def _(i, n, sy, t, w, x):
    z = sy.Function('z')
    error = sy.Sum((z(x[i], w) - t[i])**2, (i, 1, n))
    error_latex = sy.latex(error)

    (error, error_latex)
    return error, z


@app.cell
def _(error, i, n, sy, w, x, z):
    # Substitute a concrete model while keeping w constant.
    model_summand = error.function.xreplace({z(x[i], w): w[0] * x[i] + w[1]})
    model_error = sy.Sum(model_summand, (i, 1, n))
    error_for_three_points = model_error.subs(n, 3)

    (model_error, error_for_three_points)
    return


if __name__ == "__main__":
    app.run()
