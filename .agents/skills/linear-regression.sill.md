# Linear Regression — Symbolic Derivation Skill

> References: MML §9, PRML §3.1, PML §11
> Examples: `notes/01-linear-regression-mse.py`, `notes/02-ridge-regression-closed-form.py`

---

## 1. Ordinary Least Squares (MSE)

### Model

```python
x, y, w, b = symbols("x y w b", real=True)
y_hat = w * x + b
loss = Rational(1, 2) * (y_hat - y)**2
```

### Gradients

```python
grad_w = simplify(diff(loss, w))  # x*(b + w*x - y)
grad_b = simplify(diff(loss, b))  # b + w*x - y
```

\[
\frac{\partial \mathcal{L}}{\partial w} = x(\hat{y} - y),\qquad
\frac{\partial \mathcal{L}}{\partial b} = \hat{y} - y
\]

### Gradient Descent Update

\[
w \leftarrow w - \eta \cdot x(\hat{y} - y),\qquad
b \leftarrow b - \eta \cdot (\hat{y} - y)
\]

### For n training examples

```python
n = Symbol("n", integer=True, positive=True)
total_loss = Rational(1, 2) * Sum(
    (w * x_i[i] + b - y_i[i])**2, (i, 1, n)
)
```

---

## 2. Ridge Regression (ℓ₂ Regularization)

### Objective

```python
n, d = symbols("n d", integer=True, positive=True)
lam = symbols("lambda", positive=True, real=True)
X = MatrixSymbol("X", n, d)
y_vec = MatrixSymbol("y", n, 1)
w_vec = MatrixSymbol("w", d, 1)

residual = X * w_vec - y_vec
J = residual.T * residual + lam * (w_vec.T * w_vec)
```

\[
J(w) = (Xw - y)^T(Xw - y) + \lambda w^T w
\]

### Closed-Form Solution

```python
I = Identity(d)
ridge_solution = (X.T * X + lam * I).inv() * X.T * y
```

\[
w^* = (X^T X + \lambda I)^{-1} X^T y
\]

### Stationarity Condition

\[
\nabla_w J = 2X^T(Xw - y) + 2\lambda w = 0
\]

\[
(X^T X + \lambda I)w = X^T y
\]

---

## 3. Verification Pattern

Always verify equivalent gradient forms:

```python
form1 = x * (b + w*x - y)
form2 = x * (y_hat - y)
assert simplify(form1 - form2) == 0
```

---

## 4. Common Pitfalls

| Issue | Solution |
|-------|----------|
| `diff` w.r.t. `MatrixSymbol` | Use separate scalar symbols or `IndexedBase` |
| Avoiding float in sympy | Use `Rational(1, 2)` not `0.5` |
| Summation over indices | Use `Sum` with `IndexedBase` for symbolic n |
| Visualizing loss surface | Switch to numpy for numerical grid evaluation |