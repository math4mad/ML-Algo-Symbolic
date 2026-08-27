import sympy as sy
import sympy.stats as stats
data = [54, 53, 49, 61, 58]

mu = sy.Symbol("mu", real=True)
sigma = sy.Symbol("sigma", positive=True)

X = stats.Normal("X", mu, sigma)

likelihood = sy.prod(sy.stats.density(X)(value) for value in data)
likelihood = sy.factor(likelihood)

latex_expression = sy.latex(likelihood)
print(latex_expression)

likelihood