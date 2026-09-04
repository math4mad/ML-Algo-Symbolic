import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Introduction to Maximum Likelihood Estimation](https://spia.uga.edu/faculty_pages/rbakker/pols8501/MLENotes1a.pdf)

    [Derive the distribution of two Gaussian variables’ ratio with SymPy](https://niwaka-ame.github.io/articles/sympy-gaussian-quotient.html)
    """)
    return


@app.cell
def _():
    import  sympy as  sy 
    from sympy import symbols,sqrt,exp,pi
    from sympy import Symbol,simplify,log
    from sympy.stats import  Normal,density,JointRV
    x, y, mu1, mu2, sig1, sig2, r = symbols("x y \mu_1 \mu_2 \sigma_1 \sigma_2 r", real=True)
    return Normal, Symbol, density, log, simplify, sy


@app.cell
def _(latex_log_likelihood, log_likelihood, sy):
    datalatex_log_likelihood = sy.latex(log_likelihood)
    print(latex_log_likelihood)
    return


@app.cell
def _():
    # px = exp(-(x-mu1)**2 / (2 * sig1**2)) / sqrt(2 * pi) / sig1
    # py = exp(-(y-mu2)**2 / (2 * sig2**2)) / sqrt(2 * pi) / sig2
    # """
    #  pdf_2  两个正态分布的概率密度函数
    # """
    # pdf_2={'px':px,'py':py}

    # pdf_2   
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##  define Dist
    """)
    return


@app.cell
def _(Normal, Symbol, density):
    mu = Symbol("mu")
    sigma = Symbol("sigma", positive=True)

    X = Normal("x", mu, sigma)
    density(X)(50)
    return X, mu, sigma


@app.cell
def _(X, data, density, log, sigma, simplify):
    from sympy import expand
    import numpy as np
    density(X)(data[0])

    #joint_density =density(X)(data[0])*density(X)(data[1])*density(X)(data[2])
    jd=[density(X)(i) for i in data]
    joint_density=np.prod(jd)

    #simplify(joint_density)
    logL=log(joint_density)
    simplify(logL)
    logL1=logL.subs(sigma,1)

    logL1
    return logL, logL1


@app.cell
def _(logL1, mu):
    from sympy.plotting import plot
    plot(logL1, (mu, 20, 80))
    return


@app.cell
def _():
    ##  optimization 
    return


@app.cell
def _(logL, mu, sigma, simplify):
    from sympy import sin, cos, Function, diff

    dmu=diff(logL,mu)
    dmu
    dsigma=diff(logL,sigma)
    simplify(dsigma)

    expr={'dmu':simplify(dmu),'dsigma':simplify(dsigma)}
    expr
    return (expr,)


@app.cell
def _(expr, mu):
    from sympy.solvers import solve

    solve(expr["dmu"],mu)
    return (solve,)


@app.cell
def _(expr, mu, sigma, solve):



    solve(expr["dsigma"].subs(mu,55),sigma)
    return


@app.cell
def _():
    from sympy.stats import variance


    return


if __name__ == "__main__":
    app.run()
