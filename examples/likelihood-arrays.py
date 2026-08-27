import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sympy as sy
    from  sympy import log,simplify, factor,expand,diff
    from sympy import solve,collect, cancel,apart
    n = sy.Integer(5)
    i = sy.symbols("i", integer=True)
    x = sy.IndexedBase("x")

    mu = sy.Symbol("mu", real=True)
    sigma = sy.Symbol("sigma", positive=True)

    normal_density = (
        sy.exp(-(x[i] - mu)**2 / (2 * sigma**2))
        / (sy.sqrt(2 * sy.pi) * sigma)
    )

    likelihood = sy.Product(
        normal_density,
        (i, 1, n),
    )

    likelihood
    return (
        diff,
        expand,
        factor,
        likelihood,
        log,
        mu,
        sigma,
        simplify,
        solve,
        sy,
        x,
    )


@app.cell
def _(factor, likelihood):
    factor(likelihood)
    return


@app.cell
def _(expand, likelihood, log, simplify):
    simplify(expand(log(likelihood)))
    return


@app.cell
def _(expand, likelihood, log, sy, x):
    data = [54, 53, 49, 61, 58]

    data_substitutions = {
        x[index + 1]: value
        for index, value in enumerate(data)
    }

    likelihood_with_data = sy.simplify(
        likelihood.subs(data_substitutions).doit()
    )

    expand_l=expand(log(likelihood_with_data),deep=True)
    return expand_l, likelihood_with_data


@app.cell
def _(diff, likelihood_with_data, mu):
    du=diff(likelihood_with_data,mu)
    return (du,)


@app.cell
def _(du, factor, mu, solve):

    res=solve(du,mu)
    factor(res[0])
    return


@app.cell
def _(diff, expand_l, sigma):
    dsig=diff(expand_l,sigma)
    return


@app.cell
def _():
    #solve(dsig)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
