import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, exp
from scipy import linalg

from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random

np.seterr(all='print')  # ensure that we see underflow errors, etc.


class HeatEqnSolver:
    """ solves heat equation in unit square, u_t = D(u_xx + u_yy), using
    Peaceman-Rachford ADI.

    Note: Assumes that the boundary condition is 0.
        Assumes that the initial data is u(x, y, 0)=sin(pi*x)*sin(pi*y).

    Args:
        dx (float): note that 1/dx should be an int.
        dt (float): note that 1/dt should be an int.
        max_iterates (int): caps number of iterations
        D (float): problem-specific parameter

    Attributes:
        h (float): mesh spacing, dx=dy=h.
        k (float): time spacing, dt=k
        max_iterates (int): number of time steps to iterate
        D (float): diffusivity constant
        max_time (float): dt*max_iterates
        r: kD/(2h^2), a constant that reoccurs in our matrix equations.
        L (int): mesh size, our indices are integers in [0,L]
        state (array): current simulation data, shape (L+1,L+1)
        iter_count (int): number of iterates so far

    """
    def __init__(self, dx=0.01, dt=0.001, max_iterates=1000, D=1, init_data=0):
        """
        Populate essential data structures.
        dt = k, dx=dy=h. Solution will be computed up to the time dt*iterates.
        """
        self.h = dx
        self.k = dt
        self.max_iterates = max_iterates
        self.D = D
        self.max_time = dt * max_iterates
        self.r = (self.k / self.h) * (D / (2 * self.h))  # kD/2h^2

        L = round(1 /
                  dx)  # length, mesh points have coords [0, h, 2h, ..., Lh=1]
        self.L = L

        if init_data == 0:

            def initial(x, y):
                """initial data for hw4 as real function"""
                return sin(pi * x) * sin(pi * y)
        else:
            initial = init_data

        def mesh_init(i, j):
            """mesh wrapper of initial"""
            return initial(i * dx, j * dx)  # assumes dx=dy

        # populate initial data
        A = np.zeros((L + 1, L + 1))
        for i in range(1, L):
            for j in range(1, L):
                A[i, j] = mesh_init(i, j)

        self.state = A
        self.iter_count = 0

        return

    def iterate(self, n):
        """compute n iterates, enclosed for efficiency"""
        r, L, h, k = self.r, self.L, self.h, self.k

        # build the matrices we'll need
        I = np.eye(L - 1)  # diag of 1s
        J = np.eye(L - 1, k=1)  # superdiagonal of 1s
        K = np.eye(L - 1, k=-1)  # subdiagonal of 1s
        dx2 = -2 * I + J + K  # this is the second difference matrix * h^2
        A = r * dx2  # this is the matrix (kD/2)*(second difference matrix)

        # so, our equations are given by:
        # (I - A)u* = (I + A)u_old
        # (I - A)u_new = (I + A)u*
        # there is no need to recompute (I - A) and (I + A) every step.
        B = I - A
        C = I + A

        # put w=u*, u=u_old, v=u_new, and the equations become:
        # Bw = Cu
        # Bv = Cw

        # stability note: B and C are tridiagonal Toeplitz, and must be normal
        # because the subdiagonal entries equal the superdiagonal entries.
        # cf. thm 3.1 http://www.math.kent.edu/~reichel/publications/toep3.pdf

        def time_step():
            """compute one iterate, update state and iter_count"""
            for l in range(1, L):
                u = self.state[l, 1:-1]
                # solve first equation:
                Cu = C.dot(u)
                w = linalg.solve(B, Cu)
                Cw = C.dot(w)
                v = linalg.solve(B, Cw)
                self.state[l, 1:-1] = v
            return

        for _ in range(n):
            time_step()
        self.iter_count += n

    def iterate_toep(self, n):
        """toeplitz version of iterate"""
        r, L = self.r, self.L
        padding = [0] * (L - 3)
        B_data = [1 + 2 * r, -1 * r] + padding
        C_data = [1 - 2 * r, 1 * r] + padding
        C = linalg.toeplitz(C_data)

        # build the matrices we'll need:
        # put w=u*, u=u_old, v=u_new, and the equations become:
        # Bw = Cu
        # Bv = Cw

        def time_step():
            """compute one iterate, update state and iter_count"""
            for l in range(1, L):
                u = self.state[l, 1:-1]
                # solve first equation:
                Cu = C.dot(u)
                w = linalg.solve_toeplitz(B_data, Cu)
                Cw = C.dot(w)
                v = linalg.solve_toeplitz(B_data, Cw)
                self.state[l, 1:-1] = v
            return

        for _ in range(n):
            time_step()
        self.iter_count += n
        return

    def graph(self):
        """graph the current state.
        Basic 3d graph code is based on https://stackoverflow.com/a/25586869
        """
        L = self.L

        def get_data():
            """ returns the state as a list of lists to plot:
            [X, Y, Z], where X=[x_0, x_1, ...], etc.
            """
            X, Y, Z = [], [], []
            for i in range(L + 1):
                for j in range(L + 1):
                    X.append(i * self.h)
                    Y.append(j * self.h)
                    Z.append(self.state[i, j])
            return [X, Y, Z]

        X, Y, Z = get_data()
        # Z=self.state.flatten() more efficient if calling this a lot

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        norm = colors.Normalize(vmin=0, vmax=1)
        # norm2=colors.BoundaryNorm(boundaries=np.arange(0,1.01,.05), ncolors=256)
        surf = ax.plot_trisurf(X,
                               Y,
                               Z,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               norm=norm,
                               antialiased=True)
        fig.colorbar(surf)

        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.zaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_zlim(0, 1)

        ax.set_zlabel('Temp')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        ax.set_title('dt={:5.5f}, h={:5.5f}, t={:5.5f}'.format(
            self.k, self.h, self.iter_count * self.k))

        plt.show()

    def anim(self,
             iters_per_frame,
             frames,
             delay=200,
             save=False,
             title='heat_eq',
             zlim=[0, 1]):
        """Animate the dispersion. Will advance the state.

        Args:
            iters_per_frame (int): iterates this many times each frame
            frames (int):
            delay (int): gap between frames, in ms.
        """

        L = self.L
        X, Y = [], []
        for i in range(L + 1):
            for j in range(L + 1):
                X.append(i * self.h)
                Y.append(j * self.h)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.set_tight_layout(True)
        norm = colors.Normalize(vmin=0, vmax=1)
        norm = colors.BoundaryNorm(boundaries=np.arange(0, 1.01, .05),
                                   ncolors=256)

        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

        if zlim == [0, 1]:
            ax.zaxis.set_major_locator(MultipleLocator(0.5))
        else:
            ax.zaxis.set_major_locator(MaxNLocator(5))
        ax.set_zlim(zlim[0], zlim[1])

        ax.set_zlabel('Temp')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)

        def precompute_data():
            """simulate all the data first"""
            zdata = np.zeros(((L + 1)**2, frames))
            zdata[:, 0] = self.state.flatten()
            for i in range(1, frames):
                self.iterate_toep(iters_per_frame)
                zdata[:, i] = self.state.flatten()
            return zdata

        def draw(i):
            return ax.plot_trisurf(X,
                                   Y,
                                   zdata[:, i],
                                   cmap=cm.magma,
                                   linewidth=0,
                                   norm=norm,
                                   antialiased=True)

        def update(i, zdata, plot):
            """pass this method to animator.
             it draws the ith frame, then updates the state"""
            plot[0].remove()
            plot[0] = draw(i)

        zdata = precompute_data()
        plot = [draw(0)]
        animate = animation.FuncAnimation(fig,
                                          update,
                                          frames,
                                          fargs=(zdata, plot),
                                          interval=delay)

        if save:
            animate.save(title + '.gif', dpi=100, writer='imagemagick')
        plt.show()


def accuracy_check(k=0.01, steps=5):
    """Test accuracy: this should return approximately 4."""
    coarse = HeatEqnSolver(dx=0.01, dt=k)
    fine = HeatEqnSolver(dx=0.01, dt=k / 2)
    finest = HeatEqnSolver(dx=0.01, dt=k / 4)
    # Arbitrary point choice: (0.41, 0.53)
    i, j = [41, 53]  # index of point

    coarse.iterate_toep(steps)
    fine.iterate_toep(steps * 2)
    finest.iterate_toep(steps * 4)

    def v(solver):
        return solver.state[i, j]

    def R():
        return (v(coarse) - v(fine)) / (v(fine) - v(finest))

    return R()
