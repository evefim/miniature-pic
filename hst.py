#!/usr/bin/env python

from __future__ import print_function
from sys import stdout, stderr, argv, exit
import numpy as np
import matplotlib.pyplot as plt

import defs
from grid import Grid
from pml import Pml

# field update and TFSF

def update_e(grid):
    for i in range(grid.size - 1):
        A, B = grid.get_e_coeffs(i)
        grid.es[i] = A * grid.es[i] + B * (grid.bs[i] - grid.bs[i + 1])

def update_b(grid):
    for i in range(1, grid.size):
        A, B = grid.get_b_coeffs(i)
        grid.bs[i] = A * grid.bs[i] + B * (grid.es[i - 1] - grid.es[i])

def generate_b(grid, idx, t):
    x = grid.x0 + grid.dx * (idx - 0.5)
    grid.bs[idx] += grid.cdt_by_dx * defs.pulse(x, t)

def generate_e(grid, idx, t):
    x = grid.x0 + grid.dx * (idx + 1)
    grid.es[idx] += grid.cdt_by_dx * defs.pulse(x, t)


def build_plot(b_grid, idx):
    xs = [defs.dx * i for i in range(len(b_grid))]
    plt.clf()
    try:
        plt.semilogy(xs, b_grid, 'r', basey=10, nonposy='clip')
    except ValueError:
        pass
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('Bz')
    plt.ylim(1, 3e8)
    plt.savefig('{0:06d}.png'.format(idx), dpi=200)


if __name__ == '__main__':
    grid = Grid(defs.GRID_SIZE, defs.x0, defs.dx, defs.dt)
    grid.add_pml(Pml(defs.PML_SIZE, 1, grid))
    grid.add_pml(Pml(defs.GRID_SIZE - defs.PML_SIZE - 1,
            defs.GRID_SIZE - 2, grid))
    fieldgen_idx = defs.PML_SIZE + 2;

    stdout.write('iteration ')
    for t in range(3000):
        t_str = '{}'.format(t)
        stdout.write(t_str)
        stdout.flush()

        update_b(grid)
        generate_b(grid, fieldgen_idx, t * grid.dt)
        update_e(grid)
        generate_e(grid, fieldgen_idx - 1, (t + 0.5) * grid.dt)

        if t % 200 == 0:
            build_plot(grid.bs, t / 200)

        stdout.write('\b' * len(t_str))
    
    stdout.write('\n')

