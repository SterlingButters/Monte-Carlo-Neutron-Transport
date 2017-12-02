# Import the needed libraries
import numpy as np

############################################################

def get_histogram(groupBoundaries, mgXS):
    '''Given a vector of groupBoundaries of length G+1 and a vector of mgXS of length G,
    this returns two vectors, x and y, that when plotted give you a histogram.'''
    xin = groupBoundaries
    yin = mgXS
    assert len(xin) == (len(yin) + 1), 'x must be one larger than y in length.'
    xout = np.repeat(xin, 2)[1: -1]
    yout = np.repeat(yin, 2)
    return xout, yout


############################################################

def integrate(xStart, xEnd, xGrid, yGrid):
    '''Integrate y from x=xStart to x=End using the trapezoid rule.
        Expects xGrid and yGrid to be numpy arrays.
        y=f(x) is defined pointwise on yGrid with corresponding x points on xGrid.
        xStart and xEnd must be in xGrid, which must be ascendingly sorted.'''
    # Find which indexes to start and end with
    iStart = np.where(xGrid == xStart)[0][0]
    iEnd = np.where(xGrid == xEnd)[0][0]
    # Constrain x to xStart and xEnd and y to its corresponding values
    xSlice = xGrid[iStart:iEnd + 1]
    ySlice = yGrid[iStart:iEnd + 1]
    # Perform the integration with the trapezoid rule
    I = np.trapz(ySlice, xSlice)
    # Return the result
    return I


############################################################

# def run_problem():
# '''This function runs the problem'''

# Table
B = 0.01            # 1/cm
N_U238 = 6.592e-3   # atoms/barn-cm
N_U235 = 1.909e-4   # atoms/barn-cm
N_O16 = 2.757e-2    # atoms/barn-cm
N_H1 = 2.748e-2     # atoms/barn-cm
Xi_U238 = 0.00845047987286
Xi_U235 = 0.00855827595928
Xi_O16 = 0.12097997930924
Xi_H1 = 1
kT = 0.025          # eV
E_thermal = 0.1     # eV
E_fission = 50e3    # eV

############################################################ (#3)

E_0 = 20e6          # eV
E_1 = 50e3          # eV
E_2 = 1.06e3        # eV
E_3 = 55.6          # eV
E_4 = 3             # eV
E_5 = 0.1           # eV
E_6 = 1e-5          # eV
All_Energy = ([E_6, E_5, E_4, E_3, E_2, E_1, E_0])

############################################################ (#4)

# Uranium 238 Load
E238total, sigma238total = np.loadtxt('cross-sections/u-238_total.txt', skiprows=1, delimiter=',', unpack=True)
E238elastic, sigma238elastic = np.loadtxt('cross-sections/u-238_elastic.txt', skiprows=1, delimiter=',', unpack=True)
E238fission, sigma238fission = np.loadtxt('cross-sections/u-238_fission.txt', skiprows=1, delimiter=',', unpack=True)
E238gamma, sigma238gamma = np.loadtxt('cross-sections/u-238_gamma.txt', skiprows=1, delimiter=',', unpack=True)
E238chi, chi_238 = np.loadtxt('cross-sections/u-238_chi.txt', skiprows=1, delimiter=',', unpack=True)
E238nu, nu_238 = np.loadtxt('cross-sections/u-238_nu_total.txt', skiprows=1, delimiter=',', unpack=True)

# Uranium 235 Load
E235total, sigma235total = np.loadtxt('cross-sections/u-235_total.txt', skiprows=1, delimiter=',', unpack=True)
E235elastic, sigma235elastic = np.loadtxt('cross-sections/u-235_elastic.txt', skiprows=1, delimiter=',', unpack=True)
E235fission, sigma235fission = np.loadtxt('cross-sections/u-235_fission.txt', skiprows=1, delimiter=',', unpack=True)
E235gamma, sigma235gamma = np.loadtxt('cross-sections/u-235_gamma.txt', skiprows=1, delimiter=',', unpack=True)
E235chi, chi_235 = np.loadtxt('cross-sections/u-235_chi.txt', skiprows=1, delimiter=',', unpack=True)
E235nu, nu_235 = np.loadtxt('cross-sections/u-235_nu_total.txt', skiprows=1, delimiter=',', unpack=True)

# Oxygen 16 Load
E16total, sigma16total = np.loadtxt('cross-sections/o-16_total.txt', skiprows=1, delimiter=',', unpack=True)
E16elastic, sigma16elastic = np.loadtxt('cross-sections/o-16_elastic.txt', skiprows=1, delimiter=',', unpack=True)
E16gamma, sigma16gamma = np.loadtxt('cross-sections/o-16_gamma.txt', skiprows=1, delimiter=',', unpack=True)

# Hydrogen 1 Load
E1total, sigma1total = np.loadtxt('cross-sections/h-1_total.txt', skiprows=1, delimiter=',', unpack=True)
E1elastic, sigma1elastic = np.loadtxt('cross-sections/h-1_elastic.txt', skiprows=1, delimiter=',', unpack=True)
E1gamma, sigma1gamma = np.loadtxt('cross-sections/h-1_gamma.txt', skiprows=1, delimiter=',', unpack=True)

# Boron 10 Load
E10total, sigma10total = np.loadtxt('cross-sections/b-10_total.txt', skiprows=0, delimiter=',', unpack=True)
E10elastic, sigma10elastic = np.loadtxt('cross-sections/b-10_elastic.txt', skiprows=0, delimiter=',', unpack=True)
E10gamma, sigma10gamma = np.loadtxt('cross-sections/b-10_gamma.txt', skiprows=0, delimiter=',', unpack=True)
E10alpha, sigma10alpha = np.loadtxt('cross-sections/b-10_alpha.txt', skiprows=0, delimiter=',', unpack=True)


# Uranium 238 Grid
grid1 = np.array(E238total)
grid2 = np.array(E238elastic)
grid3 = np.array(E238fission)
grid4 = np.array(E238gamma)
grid5 = np.array(E238nu)
grid6 = np.array(E238chi)
y1 = sigma238total
y2 = sigma238elastic
y3 = sigma238fission
y4 = sigma238gamma
y5 = nu_238
y6 = chi_238

# Uranium 235 Grid
grid7 = np.array(E235total)
grid8 = np.array(E235elastic)
grid9 = np.array(E235fission)
grid10 = np.array(E235gamma)
grid11 = np.array(E235nu)
grid12 = np.array(E235chi)
y7 = sigma235total
y8 = sigma235elastic
y9 = sigma235fission
y10 = sigma235gamma
y11 = nu_235
y12 = chi_235

# Oxygen 16 Grid
grid13 = np.array(E16total)
grid14 = np.array(E16elastic)
grid15 = np.array(E16gamma)
y13 = sigma16total
y14 = sigma16elastic
y15 = sigma16gamma

# Hydrogen 1 Grid
grid16 = np.array(E1total)
grid17 = np.array(E1elastic)
grid18 = np.array(E1gamma)
y16 = sigma1total
y17 = sigma1elastic
y18 = sigma1gamma

# Boron 10 Grid
grid19 = np.array(E10total)
grid20 = np.array(E10elastic)
grid21 = np.array(E10gamma)
grid22 = np.array(E10alpha)
y19 = sigma10total
y20 = sigma10elastic
y21 = sigma10gamma
y22 = sigma10alpha

# np.concatenate combines all the input grids into one output grid
GridPointsTotal = np.concatenate((grid1, grid2, grid3, grid4, grid5, grid6, grid7, grid8, grid9, grid10, grid11,
                                  grid12, grid13, grid14, grid15, grid16, grid17, grid18, grid19, grid20, grid21,
                                  grid22, All_Energy))

# np.unique sorts its argument and removes any duplicates, creating our union grid
E = np.clip(GridPointsTotal, 1e-5, 20e6)
E = np.unique(E)
ThermalEnergy = np.clip(E, 0, E_thermal)
FastEnergy = np.clip(E, E_thermal, E_0)
SlowingDownEnergy = np.clip(E, E_thermal, E_fission)

# Interpolate y onto the union grid: (# 5)
UnionTotalU238 = np.interp(E, grid1, y1)
FastTotalU238 = np.interp(FastEnergy, grid1, y1)
UnionElasticU238 = np.interp(E, grid2, y2)
FastElasticU238 = np.interp(FastEnergy, grid2, y2)
UnionFissionU238 = np.interp(E, grid3, y3)
UnionGammaU238 = np.interp(E, grid4, y4)
UnionNuU238 = np.interp(E, grid5, y5)
UnionChiU238 = np.interp(E, grid6, y6)

UnionTotalU235 = np.interp(E, grid7, y7)
FastTotalU235 = np.interp(FastEnergy, grid7, y7)
UnionElasticU235 = np.interp(E, grid8, y8)
FastElasticU235 = np.interp(FastEnergy, grid8, y8)
UnionFissionU235 = np.interp(E, grid9, y9)
UnionGammaU235 = np.interp(E, grid10, y10)
UnionNuU235 = np.interp(E, grid11, y11)
UnionChiU235 = np.interp(E, grid12, y12)
ThermalChi = np.interp(ThermalEnergy, grid12, y12)
FastChi = np.interp(FastEnergy, grid12, y12)

UnionTotalO16 = np.interp(E, grid13, y13)
FastTotalO16 = np.interp(FastEnergy, grid13, y13)
UnionElasticO16 = np.interp(E, grid14, y14)
FastElasticO16 = np.interp(FastEnergy, grid14, y14)
UnionGammaO16 = np.interp(E, grid15, y15)

UnionTotalH1 = np.interp(E, grid16, y16)
FastTotalH1 = np.interp(FastEnergy, grid16, y16)
UnionElasticH1 = np.interp(E, grid17, y17)
FastElasticH1 = np.interp(FastEnergy, grid17, y17)
UnionGammaH1 = np.interp(E, grid18, y18)

UnionTotalB10 = np.interp(E, grid19, y19)
UnionElasticB10 = np.interp(E, grid20, y20)
UnionGammaB10 = np.interp(E, grid21, y21)
UnionAlphaB10 = np.interp(E, grid22, y22)