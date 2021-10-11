from abc import ABC, abstractmethod
import numpy as np



class ABC_Barrier (ABC):
    @abstractmethod
    def get_colocation_dots(self):
        pass
    @abstractmethod
    def get_discrete_approx_dots(self):
        pass
    @abstractmethod
    def get_break_off_dots(self):
        pass
    @abstractmethod
    def points_for_graph(self):
        pass


class LinearyPiecewiseBarrier(ABC_Barrier):
    def __init__(self, M: int, break_off_dots: np.ndarray):
        self.break_off_dots = break_off_dots[0] + 1j*break_off_dots[1]
        self.M = M
        self.discrete_approx_dots, self.colocation_dots = self.__creation_cloloc_discr_dots()
        
    def get_break_off_dots(self):
        return self.break_off_dots.copy()
    def get_colocation_dots(self):
        return self.colocation_dots.copy()
    def get_discrete_approx_dots(self):
        return self.discrete_approx_dots.copy()


    def __creation_cloloc_discr_dots(self):
        """
        Служебная функция для вычисления массива дискретных особ.
        и точек коллокации.
        """
        x, y = self.break_off_dots.real, self.break_off_dots.imag

        # расстояние между вершинами препядствия
        distance = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        param = np.sum(distance)
        distance = distance/param

        # вычисление кол-ва точек на каждом отрезке препядствия
        dot_on_segment = (distance * self.M).astype(int)
        dot_on_segment[-1] = self.M- np.sum(dot_on_segment[:-1])

        # создание массива дискретных особенностей  
        discrete_approx_dots = np.concatenate( [np.linspace(x[i] + 1j*y[i], x[i+1] + 1j*y[i+1] ,
                    dot_on_segment[i]+1)[:-1] for i in range(dot_on_segment.shape[0]-1)] + 
                            [np.linspace(x[-2] + 1j*y[-2], x[-1] + 1j*y[-1] , dot_on_segment[-1])])

        # создание массива точек коллокаций
        colocation_dots = (discrete_approx_dots[1:] + discrete_approx_dots[:-1])/2

        return discrete_approx_dots, colocation_dots

        # #массив индексов дли p_dots
        # self.index_p_dots = []
        # for dot in self.p_dots:
        #     self.index_p_dots.append(np.where(self.disc_dots == dot)[0][0])

    def points_for_graph(self):
        return self.break_off_dots.real, self.break_off_dots.imag




class U_FormBarrier (ABC_Barrier):
    def __init__(self, M: int):
        self.M = M
        self.__creation_cloloc_discr_dots()
        self.discrete_approx_dots, self.colocation_dots, self.z = self.__creation_cloloc_discr_dots()
        self.break_off_dots = np.array([self.discrete_approx_dots[0],self.discrete_approx_dots[-1]])


    def get_break_off_dots(self):
        return self.break_off_dots.copy()
    def get_colocation_dots(self):
        return self.colocation_dots.copy()
    def get_discrete_approx_dots(self):
        return self.discrete_approx_dots.copy()


    def __creation_cloloc_discr_dots(self):

        # break the figure into pieces
        # each piece has its own number of points
        _M = self.M+ self.M-1
        _c = _M // 3
        p0, p1, p2 = _c, _c+(_M % 3) , _c

        y_part_0 = np.linspace(0.3, 0, p0, endpoint=False)
        x_part_0 = np.array([-0.15]* p0)

        y_part_2 = np.linspace(0, 0.3, p2)
        x_part_2 = np.array([0.15]* p2)

        x_part_1 = np.linspace(-0.15, 0.15, p1, endpoint=False)
        y_part_1 = 1/0.15*x_part_1**2 -0.15

        x = np.concatenate([x_part_0,x_part_1,x_part_2])
        y = np.concatenate([y_part_0,y_part_1,y_part_2])

        z = x + 1j*y
        colocation_dots = z[1:-1:2]
        discrete_approx_dots = z[::2]

        return discrete_approx_dots,colocation_dots, z

    def points_for_graph(self):
        return self.z.real.copy(), self.z.imag.copy()

        




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    bar = U_FormBarrier(30)

    x0 = y0 = 0

    x = [x0, x0 -.25, x0 -.25, x0, x0, x0-.25]
    y = [y0, y0, y0 + .25, y0+.25 , y0+ .5 , y0+.5]
    bf_d = np.array([x,y])

    # bar = LinearyPiecewiseBarrier(20, bf_d)
    

    print(bar.get_discrete_approx_dots())

    s = bar.get_discrete_approx_dots()
    s[0] = -10

    plt.plot(bar.get_break_off_dots().real, bar.get_break_off_dots().imag)
    plt.scatter(bar.get_colocation_dots().real,bar.get_colocation_dots().imag)
    plt.scatter(bar.get_discrete_approx_dots().real,bar.get_discrete_approx_dots().imag)
    plt.show()
