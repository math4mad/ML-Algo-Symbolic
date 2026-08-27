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
    import marimo as mm


    return


@app.cell
def _():
    from sympy.abc import i, k, m, n, x
    from sympy import Sum, factorial, oo, IndexedBase, Function
    Sum(k, (k, 1, m))
    Sum(k, (k, 1, m)).doit()
    Sum(k**2, (k, 1, m))
    Sum(k**2, (k, 1, m)).doit()
    Sum(x**k, (k, 0, oo))
    Sum(x**k, (k, 0, oo)).doit()
    Sum(x**k/factorial(k), (k, 0, oo)).doit()
    return


if __name__ == "__main__":
    app.run()
