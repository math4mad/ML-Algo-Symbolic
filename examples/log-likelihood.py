import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sympy as sy

    n = sy.Integer(5)
    i = sy.symbols("i", integer=True)
    x = sy.IndexedBase("x")

    mu = sy.Symbol("mu", real=True)
    sigma = sy.Symbol("sigma", positive=True)

    log_likelihood = sy.Sum(
        -sy.log(sigma)
        - sy.log(2 * sy.pi) / 2
        - (x[i] - mu)**2 / (2 * sigma**2),
        (i, 1, n),
    )

    log_likelihood
    return i, log_likelihood, mu, sigma, sy, x


@app.cell
def _(i, mu, sigma, sy, x):
    n2 = sy.Symbol("n", integer=True, positive=True)

    log_likelihood_2 = sy.Sum(
        -sy.log(sigma)
        - sy.log(2 * sy.pi) / 2
        - (x[i] - mu)**2 / (2 * sigma**2),
        (i, 1, n2),
    )

    log_likelihood_2 
    return


@app.cell
def _(log_likelihood):
    log_likelihood
    return


if __name__ == "__main__":
    app.run()
