# Symbolic ML Formula Derivations

Use Python and `sympy` to derive, simplify, and verify machine-learning
algorithm formulas. Keep derivations symbolic before substituting data.
using marimo note style

## Setup

### 1 Environment

- Use the Conda  build a new environment `ml-sympy-env` for all  Machine Learning  sympy computing .
- The interpreter is at:
  `/Users/lunarcheung/miniconda3/envs/ml-sympy-env/bin/python`
- ⚠️  package includeding : sympy, numpy, scipy, mapplotlib, great-tables 
### 2  Reference
    ref: ./examples/  code and note 
    doc: 1.  https://docs.sympy.org/latest/ 
         2.  https://www.askpython.com/python/examples/calculus-in-python
         3.  https://courses.coe.drexel.edu/mem/memt680/topic_5/SymPy/SymPy.html
         4.  https://sbrisard.github.io/LSK/
         5.  https://skultrafast.readthedocs.io/en/stable/auto_examples/convolution.html
         5.  https://dynamics-and-control.readthedocs.io/en/latest/1_Dynamics/3_Linear_systems/Convolution.html
         6.  https://codereview.stackexchange.com/questions/174538/implementing-convolution-using-sympy
         7.  https://adamgyenge.gitlab.io/teaching/info3/2025/lec5.pdf
         8.  https://www.datacamp.com/tutorial/sympy
         9.  https://sean-fitzpatrick.github.io/CalcLabs/IntrotoSymPy.html
         10. https://agarwalla-chirag.github.io/ES114-Exposition/posts/Sympy_Stats_Module.html
         
    book : 1.[Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)
           
           2. [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/?msockid=3032e07a1dfd699a1e34f19c1c2f6842)
           3. [Mathematics for Machine Learning](https://mml-book.github.io/)

## Gaussian Distributions
   read : gaussian-dist.skill.md

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `notes/01-linear-regression-mse.py` | Linear Regression MSE gradients |
| 2 | `notes/02-ridge-regression-closed-form.py` | Ridge Regression closed-form |
| 3 | `notes/03-logistic-regression-bce.py` | Logistic Regression BCE gradients |
| 4 | `notes/04-gaussian-mle.py` | Gaussian MLE derivation |
| 5 | `notes/05-multivariate-gaussian.py` | Multivariate Gaussian (joint, marginal, conditional) |
| 6 | `notes/06-convolution.py` | Convolution symbolic derivation |
| 7 | `notes/07-error-functions.py` | Error functions (MSE, MAE, Huber) |

## Verification

Run all derivations:
```bash
python verify_derivations.py
```

CI workflow: `.github/workflows/verify.yml`

## Tag Naming Convention
-   semver: `v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v0.1.0`)
    - **MAJOR** — new notebook category or breaking formula change
    - **MINOR** — new notebook or substantial derivation added
    - **PATCH** — bug fixes, formula corrections, display / import fixes
-   use annotated tags: `git tag -a vX.Y.Z -m "description"`
-   push tags explicitly: `git push origin vX.Y.Z`

##  Rules
-   marimo note 
-   keep simple when you return token 
-   image has publish level quality
-   prepare add CI worlflow for future develop
-   `import sympy as sp` does **not** auto-load `sympy.stats`; always use
    `from sympy.stats import ...` or `import sympy.stats` before calling
    `sp.stats.*` functions (e.g. `Normal`, `MultivariateNormal`, `density`,
    `Expectation`, `marginal_distribution`).
-   avoid reusing the same variable name across cells (e.g., a SymPy summation
    index `i` will conflict with a `for i in range(...)` loop in another cell);
    use distinct names like `idx` for SymPy indices.
-   do **not** use `display()` from `IPython.display` — it is not available in
    marimo.  Instead, let the cell's last expression be the tuple of values to
    show; marimo will auto-display it. 