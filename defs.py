import math

# common constants
resolution = 32 # steps per wavelength
courant = 3 # courant number dx/c dt
COARSE_GRID_SIZE = 32 * resolution # 32 wavelengths
FINE_GRID_SIZE = 8*resolution # in coarse grid cell units
N0 = 20 # number of periods in pulse
PML_SIZE = 16
C =  2.99792458e+10
PI = 3.14159265358979
EM = 0.910938215e-27
E = -4.80320427e-10

wavelength = 0.8e-4
W0 = 2*PI*C/wavelength
REL_FIELD = -2*PI*EM*C*C/(E*wavelength)


dx = wavelength / resolution
dt = dx/(courant*C)
x0 = -COARSE_GRID_SIZE * dx / 2

DEBUG_PADDING = 2
AUX_GRID_SIZE = 1

OUTPUT_PERIOD = resolution * int(courant)
ITERATIONS = 40*OUTPUT_PERIOD + 1

# pulse description
def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def block(x, xmin, xmax):
    return (sgn(x - xmin) + sgn(xmax - x)) * 0.5

def f(t):
    return math.sin(W0*t/N0)

def form(t):
    return f(t)*f(t)*f(t*N0)*block(t, 0, N0*PI/W0)

DELAY = 16 * wavelength / C

def shape(r, t):
    return 2 * REL_FIELD * form(t + r/C - DELAY)

def pulse(x, t):
    return shape(abs(x), t)

