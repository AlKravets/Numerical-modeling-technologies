import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def modif_ax_for_barrier(axis, plot_dots : np.ndarray,
    plot_color : str = 'black' ,
    linewidth= 2,
    plot_legend : str = None,
    discr_dots : np.ndarray = None,
    color_discr_dots : str = None,
    discr_dots_legend : str = None,
    coloc_dots : np.ndarray = None,
    color_coloc_dots : str = None,
    coloc_dots_legend : str = None,
    break_off_dots : np.ndarray = None,
    color_break_off_dots :str = None,
    break_off_dots_legend : str = None
    ):

    bar_plot = axis.plot(*plot_dots, color = plot_color, label = plot_legend,linewidth= 2)

    if  not discr_dots is None:
        axis.scatter(*discr_dots, c = color_discr_dots, label = discr_dots_legend, linewidths = linewidth/2)
    
    if not coloc_dots is None:
        axis.scatter(*coloc_dots, c = color_coloc_dots, label =coloc_dots_legend, linewidths = linewidth/2)
    
    if not break_off_dots is None:
        axis.scatter(*break_off_dots, c = color_break_off_dots, label = break_off_dots_legend, linewidths = linewidth/2)


def modif_ax_for_Lv_dots(ax,Lv_dots, color : list = None,legend = None, show_line = True):
    for i in range(len(Lv_dots)):
        _c =  color[i] if not color is None else None
        if show_line:
            ax.plot(*Lv_dots[i], c = _c,marker = '.', label = legend)
        else:
            ax.scatter(*Lv_dots[i], c = _c, label = legend)


        
def modif_ax_for_pressure(fig,ax, dots, c_field,
    vmin : float = None,
    vcenter : float = None,
    vmax : float = None,
    cmap : str = 'seismic',
    levels: int = 100,
    cbar : bool = True):

    if not (vmin is None or vcenter is None or vmax is None):
        offset = mcolors.TwoSlopeNorm(vmin=vmin,
                                  vcenter=vcenter, vmax=vmax)
    else:
        offset = None

    cs = ax.contourf(*dots, c_field, levels = levels,cmap=plt.get_cmap(cmap), norm = offset)
    if cbar:
        Cbar= fig.colorbar(cs, ax = ax)

   
def modif_ax_for_vector_field(ax, dots, velocity):
    ax.quiver(*dots, *velocity)



if __name__ == "__main__":
    import barrier
    import engine
    fig, ax = plt.subplots()

    print(type(ax))
    x0=y0 = 0
    angle_koof = 3**0.5

    x1, y1 = x0, y0+ 3/4
    x2, y2 = x0-1/8 , (x0-1/8)*angle_koof +( y0 + 3/4 - angle_koof*x0)
    # dots = np.array([[x0, x1, x2], [y0, y1, y2]])
    # dots = np.array([[x2,x1,x0],[y2,y1,y0]])
    dots = np.array([[x0, x0], [y0 - 0.5, y0 + 0.5]])
    dots = np.array([[x0, x0], [y0 + 0.5, y0 - 0.5]])
    bar = barrier.LinearyPiecewiseBarrier(20, dots)

    brf = [bar.get_break_off_dots().real , bar.get_break_off_dots().imag]
    disc = [bar.get_discrete_approx_dots().real, bar.get_discrete_approx_dots().imag]
    coloc = [bar.get_colocation_dots().real, bar.get_colocation_dots().imag] 

    modif_ax_for_barrier(ax, bar.points_for_graph(), plot_color='red', plot_legend='test', discr_dots=disc, discr_dots_legend='disc', coloc_dots=coloc)
    ax.legend()
    plt.show()

    barrier_plot(bar)

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    size = 1.5
    size = 1.5
    step = 100
    
    scopes  = ((-size, 1*size),(-size, 1*size))
    x = np.linspace(scopes[0][0], scopes[0][1], step)
    y = np.linspace(scopes[1][0], scopes[1][1], step)
    
    
    fig = plt.figure()
    ax = plt.axes(xlim=scopes[0], ylim=scopes[1])



    # V_inf= np.cos(np.pi/4) + 1j*np.sin(np.pi/4)
    V_inf = 1 + 0j
    x0=y0 = 0
    angle_koof = 3**0.5

    x1, y1 = x0, y0+ 3/4
    x2, y2 = x0-1/8 , (x0-1/8)*angle_koof +( y0 + 3/4 - angle_koof*x0)
    # dots = np.array([[x0, x1, x2], [y0, y1, y2]])
    # dots = np.array([[x2,x1,x0],[y2,y1,y0]])
    dots = np.array([[x0, x0], [y0 - 0.5, y0 + 0.5]])
    dots = np.array([[x0, x0], [y0 + 0.5, y0 - 0.5]])
    bar = barrier.LinearyPiecewiseBarrier(20, dots)
    # bar = barrier.U_FormBarrier(20)
    eng = engine.Engine(bar, V_inf = V_inf)
    print(eng.Gi)
    print(f'eng.index_p_dots {eng.index_p_dots}')
    for i in range(5):
        print(f'{i+1}-----------------------------------')
        eng.update()
        
    xy = np.meshgrid(x, y)
    z = xy[0] + 1j*xy[1] 

    
    ax.plot(*bar.points_for_graph(), linewidth= 2, color = 'black')

    Lv_dots = eng.Lv_dots
    Lv_dots= np.swapaxes(np.stack([Lv_dots.real, Lv_dots.imag]), 0,1)
      
    

    modif_ax_for_Lv_dots(ax,Lv_dots, legend= 'test')
    ax.legend()

    V = eng.V_t(z)

    c = eng.C_t(z)
    print(c)
    # offset = mcolors.TwoSlopeNorm(vmin=-np.max(np.abs(c)),
    #                               vcenter=0., vmax=np.max(np.abs(c)))
    # cmap = 'seismic'
    # cs = ax.contourf(x,y,c, levels = 100,cmap=plt.get_cmap(cmap), norm = offset)
    # cbar= fig.colorbar(cs, ax = ax)
    
    

    modif_ax_for_pressure(ax,(x,y), c, vmin = -np.max(np.abs(c)), vcenter= 0, vmax = np.max(np.abs(c)))
    
    # ax.quiver(eng.Lv_dots.reshape(-1).real,eng.Lv_dots.reshape(-1).imag,V[0],V[1])
    # ax.quiver(z.real,z.imag,V[0], V[1])
    modif_ax_for_vector_field(ax, (z.real, z.imag), V)
    # ax.quiver(eng.p_dots.real,eng.p_dots.imag,V[0], V[1])
    plt.show()


    plot_barrier_velocity_pressure_at_moment((x,y), eng,
        title = 'Res'
    
    )