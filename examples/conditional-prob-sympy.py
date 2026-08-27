import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sympy as sy

    x, a = sy.symbols("x a", real=True)
    mu_x, mu_y = sy.symbols("mu_x mu_y", real=True)
    sigma_x, sigma_y = sy.symbols(
        "sigma_x sigma_y", positive=True
    )
    rho = sy.symbols("rho", real=True)

    conditional_mean = (
        mu_x + rho * sigma_x / sigma_y * (a - mu_y)
    )

    conditional_variance = sigma_x**2 * (1 - rho**2)

    conditional_density = (
        1 / sy.sqrt(2 * sy.pi * conditional_variance)
        * sy.exp(
            -(x - conditional_mean)**2
            / (2 * conditional_variance)
        )
    )

    conditional_density
    return


if __name__ == "__main__":
    app.run()
