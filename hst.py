#!/usr/bin/env python

from __future__ import print_function
from sys import stdout, stderr, argv, exit
import numpy as np
import matplotlib.pyplot as plt
import math

# constants mainly for pulse description

GRID_SIZE = 1024
C =  2.99792458e+10
PI = 3.14159265358979
EM = 0.910938215e-27
E = -4.80320427e-10
wavelength = 0.8e-4
W0 = 2*PI*C/wavelength
REL_FIELD = -2*PI*EM*C*C/(E*wavelength)
dx = wavelength / 32
dt = dx/(4*C)
cdt_by_dx = C*dt/dx

# pulse description

def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def block(x, xmin, xmax):
    return (sgn(x - xmin) + sgn(xmax - x)) * 0.5

def f(t):
    return math.sin(W0*t/20)

def form(t):
    return f(t)*f(t)*f(t*20)*block(t, 0, 20*PI/W0)

DELAY = 16 * wavelength / C

def shape(r, t):
    return 2 * REL_FIELD * form(t + r/C - DELAY)

def pulse(x, t):
    return shape(abs(x), t)

# field update and TFSF

def update_e(b_grid, e_grid, cdt_by_dx):
    for i in range(len(e_grid) - 1):
        e_grid[i] += cdt_by_dx * (b_grid[i] - b_grid[i + 1])

def update_half_b(b_grid, e_grid, cdt_by_dx):
    for i in range(1, len(b_grid)):
        b_grid[i] += 0.5 * cdt_by_dx * (e_grid[i - 1] - e_grid[i])

def generate_b(b_grid, e_grid, t):
    insertion_idx = GRID_SIZE / 2
    delta = cdt_by_dx * pulse((insertion_idx - 0.5) * dx, t)
    b_grid[insertion_idx] += delta

def generate_e(b_grid, e_grid, t):
    insertion_idx = GRID_SIZE / 2
    delta = cdt_by_dx * pulse(insertion_idx * dx, t)
    e_grid[insertion_idx - 1] += delta


def build_plot(b_grid, aux_b_grid, idx, ref_factor):
    xs = [dx * i for i in range(len(b_grid))]
    aux_bs = np.asarray([aux_b_grid[ref_factor * i] for i in range(len(b_grid))])

    plt.clf()
    try:
        plt.semilogy(xs, b_grid, 'r', basey=10, nonposy='clip')
    except ValueError:
        pass
    try:
        plt.semilogy(xs, aux_bs, 'b', basey=10, nonposy='clip')
    except ValueError:
        pass
    try:
        diff = b_grid - aux_bs
        plt.semilogy(xs, diff, 'g', basey=10, nonposy='clip')
    except ValueError:
        pass

    plt.grid()
    plt.xlabel('x')
    plt.ylabel('Bz')
    plt.ylim(1, 3e8)
    plt.savefig('{0:06d}.png'.format(idx), dpi=200)


if __name__ == '__main__':
    if len(argv) < 2:
        print('No factor passed\n\tUSAGE: ./hst.py ref_factor', file=stderr)
        exit(1)

    ref_factor = int(argv[1])
    b_grid = np.zeros(GRID_SIZE)
    e_grid = np.zeros(GRID_SIZE)
    aux_b_grid = np.zeros(ref_factor * GRID_SIZE)
    aux_e_grid = np.zeros(ref_factor * GRID_SIZE)

    HARD_SRC_IDX = GRID_SIZE / 2 + 10
    stdout.write('iteration ')
    for t in range(3000):
        t_str = '{}'.format(t)
        stdout.write(t_str)
        stdout.flush()

        update_half_b(b_grid, e_grid, cdt_by_dx)
        update_half_b(aux_b_grid, aux_e_grid, cdt_by_dx * ref_factor)
        generate_b(b_grid, e_grid, t * dt)

        aux_b_grid[HARD_SRC_IDX * ref_factor] = b_grid[HARD_SRC_IDX]

        update_e(b_grid, e_grid, cdt_by_dx)
        update_e(aux_b_grid, aux_e_grid, cdt_by_dx * ref_factor)
        generate_e(b_grid, e_grid, (t + 0.5) * dt)

        if t % 200 == 0:
            build_plot(b_grid, aux_b_grid, t / 200, ref_factor)

        update_half_b(b_grid, e_grid, cdt_by_dx)
        update_half_b(aux_b_grid, aux_e_grid, cdt_by_dx * ref_factor)

        stdout.write('\b' * len(t_str))
    
    stdout.write('\n')

