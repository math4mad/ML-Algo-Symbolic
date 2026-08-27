import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from sympy import E, Eq, Function, pde_separate, Derivative as D
    from sympy import symbols,sqrt,exp,pi,Symbol
    from sympy.abc import x, t
    from sympy.stats import Normal ,density



    return Normal, Symbol


@app.cell
def _():
    return


@app.cell
def _(Normal, Symbol):
    mu = Symbol("mu")
    sigma = Symbol("sigma", positive=True)

    X = Normal("x", mu, sigma)



    return


if __name__ == "__main__":
    app.run()
