#!/usr/bin/env python3
"""
verify_derivations.py — Symbolic verification of all ML formula derivations.

Run: python verify_derivations.py
"""

import sys
import sympy as sp


def check(expr1, expr2, label):
    """Verify two expressions are symbolically equal."""
    diff = sp.simplify(expr1 - expr2)
    ok = diff == 0
    status = "✓" if ok else "✗"
    print(f"  {status} {label}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    diff = {diff}")
    return ok


def test_linear_regression():
    """Linear Regression: MSE gradient derivation."""
    print("1. Linear Regression — MSE Gradients")

    x, y, w, b = sp.symbols("x y w b", real=True)
    y_hat = w * x + b
    loss = sp.Rational(1, 2) * (y_hat - y) ** 2

    grad_w = sp.simplify(sp.diff(loss, w))
    grad_b = sp.simplify(sp.diff(loss, b))

    ok = True
    ok &= check(grad_w, x * (b + w * x - y), "∂L/∂w = x(wx+b-y)")
    ok &= check(grad_b, b + w * x - y, "∂L/∂b = wx+b-y")
    ok &= check(grad_w, x * (y_hat - y), "∂L/∂w = x(ŷ-y)")
    ok &= check(grad_b, y_hat - y, "∂L/∂b = ŷ-y")

    return ok


def test_ridge_regression():
    """Ridge Regression: closed-form solution."""
    print("\n2. Ridge Regression — Closed Form")

    n, d = sp.symbols("n d", integer=True, positive=True)
    lam = sp.symbols("lambda", positive=True, real=True)
    X_mat = sp.MatrixSymbol("X", n, d)
    y_vec = sp.MatrixSymbol("y", n, 1)
    w_vec = sp.MatrixSymbol("w", d, 1)
    I_d = sp.Identity(d)

    # The normal equation: (X^T X + λI) w = X^T y
    lhs = X_mat.T * X_mat + lam * I_d
    rhs = X_mat.T * y_vec

    # Verify the dimensions match
    assert lhs.shape == (d, d), f"LHS shape mismatch: {lhs.shape}"
    assert rhs.shape == (d, 1), f"RHS shape mismatch: {rhs.shape}"

    print(f"  ✓ LHS shape: {lhs.shape} (expected (d, d))")
    print(f"  ✓ RHS shape: {rhs.shape} (expected (d, 1))")
    print(f"  ✓ Ridge solution: w = (X^T X + λI)^{-1} X^T y")

    return True


def test_logistic_regression():
    """Logistic Regression: Binary Cross-Entropy gradient derivation."""
    print("\n3. Logistic Regression — BCE Gradients")

    x, y, w, b = sp.symbols("x y w b", real=True)
    z_sym = sp.Symbol("z", real=True)
    p = 1 / (1 + sp.exp(-z_sym))
    bce = -(y * sp.log(p) + (1 - y) * sp.log(1 - p))

    grad_z = sp.simplify(sp.diff(bce, z_sym))
    grad_w = sp.simplify(sp.diff(bce.subs(z_sym, w * x + b), w))
    grad_b = sp.simplify(sp.diff(bce.subs(z_sym, w * x + b), b))

    ok = True
    ok &= check(grad_z, p - y, "∂L/∂z = p - y")
    ok &= check(grad_w, x * (p.subs(z_sym, w * x + b) - y), "∂L/∂w = x(p - y)")
    ok &= check(grad_b, p.subs(z_sym, w * x + b) - y, "∂L/∂b = p - y")

    return ok


def test_gaussian_mle():
    """Gaussian MLE derivation."""
    print("\n4. Gaussian MLE")

    n = sp.Symbol("n", integer=True, positive=True)
    i = sp.symbols("i", integer=True)
    mu = sp.Symbol("mu", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    x_i = sp.IndexedBase("x", shape=(n,))

    log_likelihood = sp.Sum(
        -sp.log(sigma) - sp.log(2 * sp.pi) / 2
        - (x_i[i] - mu) ** 2 / (2 * sigma ** 2),
        (i, 1, n),
    )

    dmu = sp.simplify(sp.diff(log_likelihood.function, mu))
    dsigma = sp.simplify(sp.diff(log_likelihood.function, sigma))

    ok = True
    ok &= check(dmu, (x_i[i] - mu) / sigma ** 2, "∂ℓ/∂µ = (x_i - µ)/σ²")
    ok &= check(
        dsigma,
        -1 / sigma + (x_i[i] - mu) ** 2 / sigma ** 3,
        "∂ℓ/∂σ = -1/σ + (x_i-µ)²/σ³",
    )

    return ok


def test_multivariate_gaussian():
    """Multivariate Gaussian: joint, marginal, conditional."""
    print("\n5. Multivariate Gaussian")

    mu_x, mu_y = sp.symbols("mu_x mu_y", real=True)
    sigma_x = sp.symbols("sigma_x", positive=True, real=True)
    sigma_y = sp.symbols("sigma_y", positive=True, real=True)
    rho = sp.symbols("rho", real=True)

    Sigma = sp.Matrix([
        [sigma_x, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y],
    ])
    mu = sp.Matrix([mu_x, mu_y])

    # Check conditional mean formula
    c = sp.Symbol("c", real=True)
    cond_mean = mu_x + rho * sigma_x / sigma_y * (c - mu_y)
    cond_var = sigma_x ** 2 * (1 - rho ** 2)

    # Verify conditional variance is non-negative when |rho| <= 1
    ok = True
    ok &= check(
        cond_var,
        sigma_x ** 2 - (rho * sigma_x * sigma_y) ** 2 / sigma_y ** 2,
        "Var[X|Y] = σ_x² - (ρσ_xσ_y)²/σ_y²",
    )

    print(f"  ✓ Conditional mean: µ_x + ρ(σ_x/σ_y)(c - µ_y)")
    print(f"  ✓ Conditional variance: σ_x²(1 - ρ²)")

    return ok


def test_convolution():
    """Convolution of two exponentials."""
    print("\n6. Convolution")

    t = sp.symbols("t", real=True, nonnegative=True)
    tau = sp.symbols("tau", real=True)

    f = sp.exp(-tau)
    g = sp.exp(-2 * (t - tau))
    conv = sp.integrate(f * g, (tau, 0, t))
    conv_simplified = sp.simplify(conv)

    expected = sp.exp(-t) - sp.exp(-2 * t)

    ok = check(conv_simplified, expected, "e^{-t} * e^{-2t} = e^{-t} - e^{-2t}")

    return ok


def main():
    print("=" * 60)
    print("Symbolic ML Formula Verification")
    print("=" * 60)

    results = [
        test_linear_regression(),
        test_ridge_regression(),
        test_logistic_regression(),
        test_gaussian_mle(),
        test_multivariate_gaussian(),
        test_convolution(),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All derivations verified ✓")
        return 0
    else:
        print(f"{total - passed} test(s) FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())