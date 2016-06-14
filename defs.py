import math

# common constants
resolution = 32 # steps per wavelength
courant = 5 # courant number dx/c dt
N0 = 40 # number of periods in pulse
num_wavelengths = int(N0*1.6) # length of box in wavelehgths, rounded to int value
COARSE_GRID_SIZE = num_wavelengths * resolution # 32 wavelengths
FINE_GRID_SIZE = COARSE_GRID_SIZE/4 # in coarse grid cell units
PML_SIZE = 24
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
ITERATIONS = 2*N0*OUTPUT_PERIOD + 1
DELAY = num_wavelengths/2 * wavelength / C

# pulse description
def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def block(x, xmin, xmax):
    return (sgn(x - xmin) + sgn(xmax - x)) * 0.5

def f(t):
    return math.sin(W0*t/N0)

def form(t):
    return f(t)*f(t)*f(t*N0)*block(t, 0, N0*PI/W0)

def shape(r, t):
    return form(t + r/C - DELAY)

def pulse(x, t):
    return shape(abs(x), t)

