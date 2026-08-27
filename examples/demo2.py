import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    return mo, multivariate_normal, np, plt


@app.cell(hide_code=True)
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


@app.cell(disabled=True)
def _(multivariate_normal, np, plt, rho, slider_X, slider_Y):
    mu1,mu2,coefs=slider_X.value,slider_Y.value,rho.value
    sig1,sig2=1.0,1.0
    Sig=np.array([[sig1,sig1*sig2*coefs],[sig1*sig2*coefs,sig2]])
    # def get_data():
    #     x, y = np.mgrid[-3:3:.01, -3:3:.01]
    #     pos = np.dstack((x, y))
    #     rv = multivariate_normal([mu1,mu2],Sig)
    #     data=rv.pdf(pos)
    #     #dt=pd.DataFrame({"x":x,"y":y,"data":data})
    #     return  x,y,pos,rv

    def  normal_dist():
         x, y = np.mgrid[-3:3:.01, -3:3:.01]
         pos = np.dstack((x, y))
         rv = multivariate_normal([mu1,mu2],Sig)
         return x,y, pos,rv
    x,y,pos,rv= normal_dist()
    def  plot_plt(x,y,pos,rv):
         fig = plt.figure()
         ax = fig.add_subplot(111)
         return ax.contourf(x, y, rv.pdf(pos))

    plot_plt(x,y,pos,rv)
    return mu1, mu2, sig1, sig2


@app.cell(disabled=True)
def _():
    #import plotnine as pl9
    #from  plotnine import ggplot as gg
    import pandas as pd
    # def get_data():
    #     x, y = np.mgrid[-3:3:.01, -3:3:.01]
    #     pos = np.dstack((x, y))
    #     rv = multivariate_normal([mu1,mu2],Sig)
    #     data=rv.pdf(pos)
    #     #dt=pd.DataFrame({"x":x,"y":y,"data":data})
    #     return  x,y,pos,rv,data

    # data=get_data()
    # data
    return


@app.cell(disabled=True)
def _(mu1, mu2, sig1, sig2):
    from scipy.stats import norm
    distx=norm(mu1,sig1)
    disty=norm(mu2,sig2)

    # def  plot2(x,y,pos,rv):
    #      fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    #      axs[0].plot(x,distx.pdf(x))
    #      axs[1].contourf(x, y, rv.pdf(pos))
    #      axs[2].plot(disty.pdf(y),y)
    #      return plt.show()


    # plot2(x,y,pos,rv)
    return


@app.cell(hide_code=True)
def _(mo, rho, slider_X, slider_Y):
    mo.md(
        f"""


        -  X-Axis   $\mu_x$: {slider_X}   
        -  Y-Axis   $\mu_y$: {slider_Y}   
        -  Coefs  $\\rho$: {rho}  


        """
    )   
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
