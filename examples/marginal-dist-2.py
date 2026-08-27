import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal, norm

    return mo, multivariate_normal, norm, np, plt


@app.cell
def _(mo):
    slider_X = mo.ui.slider(start=-2, stop=2, step=0.1,show_value=True)
    slider_Y = mo.ui.slider(start=-2, stop=2, step=0.2,show_value=True)
    rho      =     mo.ui.slider(start=-1, stop=1, step=0.1,show_value=True)

    mo.md(
            f"""
   

            -  X-Axis   $\mu_x$: {slider_X}   
            -  Y-Axis   $\mu_y$: {slider_Y}   
            -  Coefs  $\\rho$: {rho}  


            """
    )   
    return rho, slider_X, slider_Y


@app.cell
def _(np, rho, slider_X, slider_Y):
    sig1,sig2=1.0,1.0
    mu1,mu2,coefs=slider_X.value,slider_Y.value,rho.value
    Sig=np.array([[sig1,sig1*sig2*coefs],[sig1*sig2*coefs,sig2]])
    return


@app.cell
def _(multivariate_normal):
    def bivariate_normal(μ_x, μ_y, σ_x, σ_y, ρ):
        cov = [[σ_x**2, ρ * σ_x * σ_y],
               [ρ * σ_x * σ_y, σ_y**2]]
        return multivariate_normal([μ_x, μ_y], cov)

    return (bivariate_normal,)


@app.cell
def _(bivariate_normal, norm, np, plt):
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    pos = np.dstack((X_mesh, Y_mesh))
    data_xz=norm(0,1).pdf(y_grid)
    data_yz=norm(0,1).pdf(y_grid)
    u = bivariate_normal(0, 0, 1, 1, 0.6)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X_mesh, Y_mesh, u.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    plt.show()
    return X_mesh, Y_mesh, pos


@app.cell
def _(X_mesh, Y_mesh, bivariate_normal, plt, pos):
    def  plot_diff_coef():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for ax, ρ in zip(axes, (-0.8, 0.0, 0.8)):
            u = bivariate_normal(0, 0, 1, 1, ρ)
            ax.contour(X_mesh, Y_mesh, u.pdf(pos), levels=6, cmap='viridis')
            ax.set_title(rf'$\rho={ρ}$')
            ax.set_xlabel('x')
            ax.set_aspect('equal')
            axes[0].set_ylabel('y')
        plt.show()

    plot_diff_coef()

    return


if __name__ == "__main__":
    app.run()
