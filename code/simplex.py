from typing import Callable, Tuple, Union

import numpy as np
import scipy as sp
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as animation


def order(x: np.ndarray, ordering: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(ordering)
    return x[indices], ordering[indices]


def NelderMead(
    fun: Callable,
    x0: np.ndarray,
    maxiter: Union[int, None] = None,
    initial_simplex: Union[np.ndarray, None] = None
):
    if x0.ndim != 1:
        raise ValueError(f'Expected 1D array, got {x0.ndim}D array instead')

    # initialize simplex
    if initial_simplex is not None:
        if initial_simplex.ndim != 2:
            raise ValueError(f'Expected 2D array, got {x0.ndim}D array instead')
        x = initial_simplex.copy()
        n = x[0].size
    else:
        h = lambda x: (x[0][x[1]] != 0) * (0.05 - 0.00025) + 0.00025
        n = x0.size
        x = np.array([x0 + h([x0, i]) * e for i, e in enumerate(np.identity(n))] + [x0])

    if maxiter is None:
        maxiter = 200 * n

    # parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    # order
    fx = np.array(list(map(fun, x)))
    x, fx = order(x, fx)

    # centroid
    xo = np.mean(x[:-1], axis=0)
    n_inv = 1 / n

    for _ in range(maxiter):
        fx1 = fx[0]
        fxn = fx[-2]
        fxmax = fx[-1]
        xmax = x[-1]

        xr = xo + alpha * (xo - xmax)
        fxr = fun(xr)

        if fx1 <= fxr and fxr < fxn:
            # reflect
            x[-1] = xr
            fx[-1] = fun(xr)
            x, fx = order(x, fx)
            xo = xo + n_inv * (xr - x[-1])

        elif fxr < fx1:
            xe = xo + gamma * (xo - xmax)
            fxe = fun(xe)
            if fxe < fxr:
                # expand
                x = np.append(xe.reshape(1, -1), x[:-1], axis=0)
                fx = np.append(fxe, fx[:-1])
                xo = xo + n_inv * (xe - x[-1])
            else:
                # reflect
                x = np.append(xr.reshape(1, -1), x[:-1], axis=0)
                fx = np.append(fxr, fx[:-1])
                xo = xo + n_inv * (xr - x[-1])

        else:
            if fxr > fxmax:
                xc = xo + rho * (xmax - xo)
            else: 
                xc = xo + rho * (xr - xo)
                fxmax = fxr
            if fun(xc) < fxmax:
                # contract
                x[-1] = xc
                fx[-1] = fun(xc)
                x, fx = order(x, fx)
                xo = xo + n_inv * (xc - x[-1])
            else:
                # shrink
                x[1:] = (1 - sigma) * x[0] + sigma * x[1:]
                fx[1:] = np.array(list(map(fun, x[1:])))
                x, fx = order(x, fx)
                xo = np.mean(x[:-1], axis=0)

    return x, fx


def constructGIF(x, xmin, xmax, ymin, ymax, xx, yy, vals):
    # clear the current axes
    plt.cla()
    
    # set x-axis and y-axis
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.hlines(0, xmin=xmin, xmax=xmax, colors='gray')
    plt.vlines(0, ymin=ymin, ymax=ymax, colors='gray')
    
    # set aspect
    plt.gca().set_aspect('equal', adjustable='box')
    
    # draw filled contour
    plt.contourf(xx, yy, vals, 100, cmap='Blues')
    plt.contour(xx, yy, vals, 50, lw=1)
    
    # draw triangle
    plt.axes().add_patch(pat.Polygon(x, ec='k', fc='m', alpha=0.2))
    
    # draw three vertices
    plt.scatter(x[:, 0], x[:, 1], color=['y', 'g', 'r'], s=20)