import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _():
    import marimo as mo
    y_slider=mo.ui.slider(start=-3.0,
    stop=3.0,
    step=0.1,
    value=0.0,
    label="Fixed value of y",
    )
  
    return (y_slider,)


@app.cell
def _(np, plt, y_slider):
    mu_x=0.0
    mu_y = 0.0
    sigma_x = 1.0
    sigma_y = 1.4
    rho = 0.7

    fixed_y = y_slider.value
    conditional_mean = (
        mu_x + rho * sigma_x / sigma_y * (fixed_y - mu_y)
    )
    conditional_variance = sigma_x**2 * (1 - rho**2)
    conditional_std = np.sqrt(conditional_variance)

    x_values = np.linspace(-4, 4, 250)
    y_values = np.linspace(-4, 4, 250)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    covariance_matrix = np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2],
    ])
    inverse_covariance = np.linalg.inv(covariance_matrix)
    centered_grid = np.stack(
        [grid_x - mu_x, grid_y - mu_y],
        axis=-1,
    )
    joint_density = (
        np.exp(
            -0.5
            * np.einsum(
                "...i,ij,...j->...",
                centered_grid,
                inverse_covariance,
                centered_grid,
            )
        )
        / (2 * np.pi * np.sqrt(np.linalg.det(covariance_matrix)))
    )

    conditional_density = (
        np.exp(
            -0.5
            * ((x_values - conditional_mean) / conditional_std) ** 2
        )
        / (np.sqrt(2 * np.pi) * conditional_std)
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )

    contour = axes[0].contourf(
        grid_x,
        grid_y,
        joint_density,
        levels=20,
        cmap="viridis",
    )
    axes[0].axhline(
        fixed_y,
        color="white",
        linewidth=2,
        label=fr"$y={fixed_y:.1f}$",
    )
    axes[0].set(
        title="Joint density $p(x,y)$",
        xlabel="$x$",
        ylabel="$y$",
    )
    axes[0].legend()

    axes[1].plot(
        x_values,
        conditional_density,
        color="#e45756",
        linewidth=3,
    )
    axes[1].axvline(
        conditional_mean,
        color="#2f4858",
        linestyle="--",
        label=fr"$\\mu_{{X|Y}}={conditional_mean:.2f}$",
    )
    axes[1].set(
        title=fr"$p(x \\mid y={fixed_y:.1f})$",
        xlabel="$x$",
        ylabel="density",
    )
    axes[1].legend()
    figure.colorbar(contour, ax=axes[0], label="density")
    #figure.tight_layout()

    figure
    return


if __name__ == "__main__":
    app.run()
