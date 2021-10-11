import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import barrier
import engine
import graphics

def barrier_plot(
    barrier : barrier.ABC_Barrier,
    fig_size: tuple = None,
    grid: bool = False,
    title: str = None,
    xlabel: str = '$x$',
    ylabel : str = '$y$',
    x_lim: list = [],
    y_lim: list = [],
    plot_color : str = 'black' ,
    plot_legend : str = None,
    show_discr_dots : bool = False,
    color_discr_dots : str = None,
    discr_dots_legend : str = None,
    show_coloc_dots : bool = False,
    color_coloc_dots : str = None,
    coloc_dots_legend : str = None,
    show_break_off_dots : bool = False,
    color_break_off_dots :str = None,
    break_off_dots_legend : str = None
    ):
    
    fig, ax = plt.subplots(figsize = fig_size)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)
    if x_lim and y_lim:
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

    disc = None
    coloc = None
    brf = None

    if show_discr_dots:
        disc = [barrier.get_discrete_approx_dots().real,
                barrier.get_discrete_approx_dots().imag]
    if show_coloc_dots:
        coloc = [barrier.get_colocation_dots().real,
                barrier.get_colocation_dots().imag]
    if show_break_off_dots:
        brf = [barrier.get_break_off_dots().real,
                barrier.get_break_off_dots().imag]

    graphics.modif_ax_for_barrier(ax, barrier.points_for_graph(),
        plot_color = plot_color, plot_legend = plot_legend,
        discr_dots= disc, color_discr_dots = color_discr_dots, discr_dots_legend = discr_dots_legend,
        coloc_dots= coloc, color_coloc_dots = color_coloc_dots, coloc_dots_legend= coloc_dots_legend,
        break_off_dots= brf, color_break_off_dots= color_break_off_dots, break_off_dots_legend = break_off_dots_legend)
    ax.legend(loc ='upper left')
    plt.show()
    

def plot_barrier_velocity_pressure_at_moment(dots, eng : engine.Engine, 
    fig_size: tuple = None,
    grid: bool = False,
    title: str = None,
    xlabel: str = '$x$',
    ylabel : str = '$y$',
    x_lim: list = [],
    y_lim: list = [],
    axis_equal = False,
    show_Lv_dots : bool = True,
    Lv_dots_color : list = None,
    show_Lv_dots_line = True,
    Lv_dots_legend : str = None,
    show_pressure : bool = True,
    norm: bool = True,
    cmap : str = 'seismic',
    levels: int = 100,
    cbar : bool = True,
    show_velocity : bool = True,
    show_barrier : bool = True,
    barrier_color : str = 'black',
    barrier_linewidth: int = 2,
    barrier_legend : str = None,
    show_discr_dots : bool = False,
    color_discr_dots : str = None,
    discr_dots_legend : str = None,
    show_coloc_dots : bool = False,
    color_coloc_dots : str = None,
    coloc_dots_legend : str = None,
    show_break_off_dots : bool = False,
    color_break_off_dots :str = None,
    break_off_dots_legend : str = None
    ):

    fig, ax = plt.subplots(figsize = fig_size)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)
    if x_lim and y_lim:
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
    if axis_equal:
        ax.axis('equal')

    x,y = dots
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1]
    V = eng.V_t(z)
    c = eng.C_t(z)
    if show_pressure:
        vmin, vcenter, vmax = None, None, None
        if norm:
            vmin = -np.max(np.abs(c))
            vcenter = 0
            vmax = np.max(np.abs(c))
        graphics.modif_ax_for_pressure(fig, ax, dots, c, vmin = vmin, vcenter= vcenter, vmax = vmax, cmap = cmap, levels= levels, cbar=cbar)
    if show_velocity:
        graphics.modif_ax_for_vector_field(ax, xy, V)

    if show_Lv_dots:
        Lv_dots = eng.Lv_dots
        Lv_dots= np.swapaxes(np.stack([Lv_dots.real, Lv_dots.imag]), 0,1)
        graphics.modif_ax_for_Lv_dots(ax,Lv_dots, color = Lv_dots_color, legend = Lv_dots_legend, show_line= show_Lv_dots_line)



    barrier = eng.bar

    disc = None
    coloc = None
    brf = None

    if show_discr_dots:
        disc = [barrier.get_discrete_approx_dots().real,
                barrier.get_discrete_approx_dots().imag]
    if show_coloc_dots:
        coloc = [barrier.get_colocation_dots().real,
                barrier.get_colocation_dots().imag]
    if show_break_off_dots:
        brf = [barrier.get_break_off_dots().real,
                barrier.get_break_off_dots().imag]
    if show_barrier:
        graphics.modif_ax_for_barrier(ax, barrier.points_for_graph(),
            plot_color = barrier_color, plot_legend = barrier_legend, linewidth= barrier_linewidth,
            discr_dots= disc, color_discr_dots = color_discr_dots, discr_dots_legend = discr_dots_legend,
            coloc_dots= coloc, color_coloc_dots = color_coloc_dots, coloc_dots_legend= coloc_dots_legend,
            break_off_dots= brf, color_break_off_dots= color_break_off_dots, break_off_dots_legend = break_off_dots_legend)


    ax.legend(loc ='upper left')
    plt.show()


#
x0, y0 = 0, 0 

# plate
dots_plate = np.array([[x0, x0], [y0 + 0.5, y0 - 0.5]])
# 2
dots_2 = np.array([[x0, x0 -.25, x0 -.25, x0, x0, x0-.25],
            [y0, y0, y0 + .25, y0+.25 , y0+ .5 , y0+.5]])
# 3
dots_3 = np.array([[x0, x0+.25, x0, x0+0.25, x0],
            [y0, y0+ .25, y0 + .5 , y0+ .75, y0 + 1]])
# 5
dots_5 = np.array([[x0, x0+0.5, x0+0.5, x0, x0, x0+0.5],
                [y0, y0, y0+0.5, y0+0.5, y0+0.75, y0+0.75]])



def main():
    
    count_updates = 100

    x_lim = [-1,4]
    y_lim = [-1.5,1.5]
    xsteps = 100
    ysteps = 50

    x = np.linspace(*x_lim, xsteps)
    y = np.linspace(*y_lim, ysteps)

    V_inf = 1 + 0j

    M = 20

    bar = barrier.LinearyPiecewiseBarrier(M,dots_plate)
    # bar = barrier.U_FormBarrier(M)    

    eng = engine.Engine(bar, V_inf = V_inf, delta= 5*10** -2)

    for i in range(count_updates):
        print(f'{i+1}-----------------------------------')
        eng.update()

    plot_barrier_velocity_pressure_at_moment(
        (x,y),
        eng,
        fig_size= (12,7),
        grid=False,
        title = f'Time {eng.t:.2f}, count updates {eng.n}',
        x_lim = x_lim,
        y_lim = y_lim,
        axis_equal=False,
        show_Lv_dots=True,
        show_Lv_dots_line = False,
        Lv_dots_legend= '$L_v$ contour points',
        show_pressure=True,
        norm = True,
        cmap = 'seismic',
        levels = 100,
        cbar = True,
        show_velocity= True,
        show_barrier=True,
        barrier_legend= 'Barrier contour',
        show_discr_dots=False,
        discr_dots_legend= 'discrete approximation dots',
        show_coloc_dots= False,
        coloc_dots_legend= 'colocation dots',
        show_break_off_dots=False,
        break_off_dots_legend='break off dots',
    )


if __name__ == "__main__":
    main()


    # x_lim = [-1,1.5]
    # y_lim = [-1.5,1.5]
    # xsteps = 100
    # ysteps = 50

    # x = np.linspace(*x_lim, xsteps)
    # y = np.linspace(*y_lim, ysteps)

    # V_inf = 1 + 0j

    # M = 20

    # bar = barrier.LinearyPiecewiseBarrier(M,dots_2)
    # # bar = barrier.U_FormBarrier(M)    
    # barrier_plot(bar, 
    # grid= True,
    # x_lim=(-0.5,0.5),
    # y_lim=(-0.25,0.75),
    # title= '2 form barrier',
    # show_discr_dots= True,
    # discr_dots_legend= 'discrete approximation dots',
    # show_coloc_dots= True,
    # coloc_dots_legend= 'colocation dots',
    # show_break_off_dots=True,
    # break_off_dots_legend='break off dots',
    # )