import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sympy as sy

    from sympy import (
        MatrixSymbol,
        BlockMatrix,
        symbols,
        Identity,
        ZeroMatrix,
        block_collapse,
        Matrix,
        Symbol,
    )
    from sympy.stats import MultivariateNormal, Normal, density, given

    return (
        Matrix,
        MatrixSymbol,
        MultivariateNormal,
        Normal,
        Symbol,
        density,
        sy,
        symbols,
    )


@app.cell
def _(Matrix, Symbol, symbols):

    a,b=symbols('a,b')
    mu1,mu2=symbols('mu_1,mu_2')
    mu=Matrix([mu1,mu2])
    y1,y2=symbols('y_1,y_2')
    sig11=Symbol('Sigma_11')
    sig12=Symbol('Sigma_12')
    sig21=Symbol('Sigma_21')
    sig22=Symbol('Sigma_22')
    Sigma=Matrix([[sig11,sig12],[sig21,sig22]])
    Sigma 

    (mu1,mu2,mu, y1,y2,Sigma)
    return Sigma, mu, mu1, mu2, sig11, sig12, sig21, sig22, y1


@app.cell
def _(MatrixSymbol, symbols):
    n = symbols('n', integer=True, positive=True)
    obs = MatrixSymbol('obs', n, 1)


    return (obs,)


@app.cell
def _(MultivariateNormal, Sigma, density, mu, obs):
    X = MultivariateNormal('X', mu, Sigma)
    density(X)(obs)
    return


@app.cell
def _(Normal, mu1, mu2, sig11, sig12, sig21, sig22, sy, y1):
    # Step 1: condition the bivariate normal on Y = c.
    c = sy.Symbol('c', real=True)

    # `given` cannot condition indexed components of MultivariateNormal
    # in the current SymPy API, so use Murphy's covariance formula.
    mu_y1_given_y2 = mu1 + sig12 / sig22 * (c - mu2)
    var_y1_given_y2 = sig11 - sig12 * (1 / sig22) * sig21

    conditional_density_y1_given_y2 = (
        sy.exp(
            -(y1 - mu_y1_given_y2)**2
            / (2 * var_y1_given_y2)
        )
        / sy.sqrt(2 * sy.pi * var_y1_given_y2)
    )

    conditional_expression = (
        sy.Eq(
            sy.Symbol('Y_1 | Y_2 = c'),
            Normal(
                'Y_1_given_Y_2',
                mu_y1_given_y2,
                sy.sqrt(var_y1_given_y2),
            ),
        ),
        conditional_density_y1_given_y2,
    )

    conditional_expression
    return


if __name__ == "__main__":
    app.run()
