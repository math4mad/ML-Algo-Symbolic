# Gaussian Distribution — Symbolic Derivation Skill

> References: PRML §2.3, MML §6.5, PML §3.2
> SymPy API: `sympy.stats.Normal`, `MultivariateNormal`, `density`, `marginal_distribution`

---

## 1. Univariate Gaussian

### Setup (from `examples/normal-dist-sym.py`, `examples/deduce-mle-gaussian.py`)

```python
from sympy.stats import Normal, density, cdf, Expectation, variance, P
from sympy import symbols, sqrt, exp, pi, simplify, Symbol

mu = Symbol("mu", real=True)
sigma = Symbol("sigma", positive=True)
X = Normal("X", mu, sigma)

# PDF
density(X)(x)  # exp(-(-mu + x)^2/(2*sigma^2))/(sqrt(2)*sqrt(pi)*sqrt(sigma))

# CDF — standard normal
Norm = Normal('Norm', 0, 1)
cdf(Norm)  # Lambda(_z, erf(sqrt(2)*_z/2)/2 + 1/2)

# Probability in interval
P(Norm > 0, Norm < 2)  # exact erf expression

# Moments
Expectation(X)         # mu
variance(X)            # sigma^2
```

### Key formulas

\[
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\]

\[
\Phi(z) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right]
\]

---

## 2. MLE for Gaussian Parameters

### From `examples/deduce-mle-gaussian.py`, `examples/likelihood-arrays.py`, `examples/log-likelihood.py`

**Log-likelihood** for n i.i.d. observations:

```python
n = Symbol("n", integer=True, positive=True)
i = symbols("i", integer=True)
x_i = IndexedBase("x", shape=(n,))

log_likelihood = Sum(
    -log(sigma) - log(2*pi)/2 - (x_i[i] - mu)**2 / (2*sigma**2),
    (i, 1, n),
)
```

**Score equations:**

\[
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0
\quad\Rightarrow\quad
\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i
\]

\[
\frac{\partial \ell}{\partial \sigma} = -\frac{n}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^n (x_i - \mu)^2 = 0
\quad\Rightarrow\quad
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2
\]

---

## 3. Bivariate Gaussian

### From `examples/Statistics/mvnormal.ipynb`, `examples/Statistics/gaussian-dist-sympy.ipynb`

```python
from sympy.stats import MultivariateNormal, density, marginal_distribution
from sympy import Matrix, symbols

mu_x, mu_y = symbols("mu_x mu_y", real=True)
sigma_x = symbols("sigma_x", positive=True)
sigma_y = symbols("sigma_y", positive=True)
rho = symbols("rho", real=True)

mu = [mu_x, mu_y]
Sigma = Matrix([
    [sigma_x, rho * sigma_x * sigma_y],
    [rho * sigma_x * sigma_y, sigma_y],
])

m = MultivariateNormal("X", mu, Sigma)
x, y = symbols("x y")

# Joint density
density(m)(x, y)  # full bivariate Gaussian expression

# Marginal of X (first component)
marginal_distribution(m, m[0])(y)  # N(mu_x, sigma_x)
```

### Joint density

\[
p(x,y) = \frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}}
\exp\left(-\frac{1}{2(1-\rho^2)}\left[
\frac{(x-\mu_x)^2}{\sigma_x^2} +
\frac{(y-\mu_y)^2}{\sigma_y^2} -
\frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y}
\right]\right)
\]

---

## 4. Conditional Gaussian

### From `examples/posterior-conditional.py`, `examples/conditional-dist.py`

**Important:** `sympy.stats.given` cannot directly condition components of `MultivariateNormal` in the current API. Use **Murphy's formula** instead:

```python
c = Symbol("c", real=True)

mu_y1_given_y2 = mu1 + sig12 / sig22 * (c - mu2)
var_y1_given_y2 = sig11 - sig12 * (1 / sig22) * sig21

conditional_density = (
    exp(-(y1 - mu_y1_given_y2)**2 / (2 * var_y1_given_y2))
    / sqrt(2 * pi * var_y1_given_y2)
)
```

### Formula

\[
X_1 \mid X_2 = x_2 \sim \mathcal{N}\!\left(
\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2),\;
\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\right)
\]

For the bivariate case with correlation $\rho$:

\[
X \mid Y = c \;\sim\; \mathcal{N}\!\left(
\mu_x + \rho\frac{\sigma_x}{\sigma_y}(c - \mu_y),\;
\sigma_x^2(1 - \rho^2)
\right)
\]

---

## 5. n-Dimensional Multivariate Gaussian

### From `examples/Statistics/gaussian-dist-sympy.ipynb`

```python
n = symbols('n', integer=True, positive=True)
Sg = MatrixSymbol('Sigma', n, n)
mu = MatrixSymbol('mu', n, 1)
obs = MatrixSymbol('obs', n, 1)

X = MultivariateNormal('X', mu, Sg)
density(X)(obs)  # generic n-dim PDF
```

\[
p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}}
\exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
\]

---

## 6. Covariance Matrix Properties

### From `examples/Statistics/define-covariance-matrix.ipynb`

```python
from sympy.stats import VarianceMatrix
from sympy.stats.rv import RandomMatrixSymbol
from sympy import symbols, MatrixSymbol

k = symbols("k")
A = MatrixSymbol("A", k, k)
X = RandomMatrixSymbol("X", k, 1)

VarianceMatrix(X)           # Var(X)
VarianceMatrix(A * X).expand()  # A * Var(X) * A^T
```

---

## 7. Convolution of Gaussians

### From `examples/convolution.ipynb`

\[
\mathcal{N}(\mu_1, \sigma_1^2) * \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
\]

The convolution of two independent Gaussians adds means and variances.

---

## 8. Common Pitfalls

| Issue | Solution |
|-------|----------|
| `given(m, Eq(y, c))` fails on `MultivariateNormal` | Use Murphy's formula directly |
| `density` returns messy unsimplified expression | Apply `simplify()` |
| `MatrixSymbol` indices not resolving | Use explicit `Matrix` for small dimensions |
| `diff` w.r.t. composite expression | Use a separate `Symbol` and substitute |