# Import the needed libraries
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys


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

# plt.figure()
# plt.plot(E, UnionChiU235, label='U235')
# plt.plot(E, UnionChiU238, label='U238')
# plt.legend()
# plt.show()

############################################################ (#4)

# ST = (N_U238 * UnionTotalU238 + N_U235 * UnionTotalU235 + N_O16 * UnionTotalO16 + N_H1 * UnionTotalH1)
# FastST = (N_U238 * FastTotalU238 + N_U235 * FastTotalU235 + N_O16 * FastTotalO16 + N_H1 * FastTotalH1)
# SigmaGamma = (N_U238 * UnionGammaU238 + N_U235 * UnionGammaU235 + N_O16 * UnionGammaO16 + N_H1 * UnionGammaH1)
# SigmaScatter = (
# N_U238 * UnionElasticU238 + N_U235 * UnionElasticU235 + N_O16 * UnionElasticO16 + N_H1 * UnionElasticH1)
# FastSigmaScatter = (
# N_U238 * FastElasticU238 + N_U235 * FastElasticU235 + N_O16 * FastElasticO16 + N_H1 * FastElasticH1)
# SNF = (N_U238 * UnionFissionU238 * UnionNuU238 + N_U235 * UnionFissionU235 * UnionNuU235)
# SigmaFission = (N_U238 * UnionFissionU238 + N_U235 * UnionFissionU235)
# SA = (ST - SigmaScatter)
# FastSA = (FastST - FastSigmaScatter)
# SigmaScatterH1 = (N_H1 * UnionElasticH1)
# FastSigmaScatterH1 = (N_H1 * FastElasticH1)
# D = 1 / (3 * ST)
# FastD = 1 / (3 * FastST)
# Xi = ((N_U238 * UnionElasticU238 * Xi_U238 + N_U235 * UnionElasticU235 * Xi_U235 + N_O16 * UnionElasticO16 * Xi_O16 + N_H1 * UnionElasticH1 * Xi_H1) /
#       (N_U238 * UnionElasticU238 + N_U235 * UnionElasticU235 + N_O16 * UnionElasticO16 + N_H1 * UnionElasticH1))
#
# ############################################################ (#5)
#
# p_old = (np.exp(-integrate(E_thermal, E_0, E, SA / (E * Xi * (D * B ** 2 + ST)))))
# p_new = (np.exp(-integrate(E_thermal, E_0, E, SA / (E * (D * B ** 2 + ST)))))
# print('p_old:', p_old)
# print('p_new:', p_new)
#
# ############################################################ (#6)
#
# f_ssunshielded = 1
# f_ssNRIM = 1 / (SA + SigmaScatterH1 + D * B ** 2)
# f_ssNR = 1 / (ST + D * B ** 2)
#
# f_ssNRIMFast = 1 / (FastSA + FastSigmaScatterH1 + FastD * B ** 2)
# f_ssNRFast = 1 / (FastST + FastD * B ** 2)
#
# # c_fission*Xi(E_fission) = 1/E_fission
# # c_thermal*E_th/kT*exp(-E_th/kT)=(1/E_th+c_fission*Xi(E_th))*f_ss(E_th)
#
# c_fission = 1 / (E_fission * UnionChiU235[np.where(E == E_fission)])
# c_thermalunshielded = (((1 / E_thermal + c_fission * UnionChiU235[np.where(E == E_thermal)]) * f_ssunshielded) / (
# E_thermal / kT * np.exp(-E_thermal / kT)))
#
# c_thermalNRIM = (((1 / E_thermal + c_fission * UnionChiU235[np.where(E == E_thermal)]) * f_ssNRIM[np.where(E == E_thermal)]) / (
# E_thermal / kT * np.exp(-E_thermal / kT)))
# c_thermalNR = (
# ((1 / E_thermal + c_fission * UnionChiU235[np.where(E == E_thermal)]) * f_ssNR[np.where(E == E_thermal)]) / (
# E_thermal / kT * np.exp(-E_thermal / kT)))
#
# f_unshieldedthermal = c_thermalunshielded * ThermalEnergy / kT * np.exp(-ThermalEnergy / kT)
# f_unshieldedfast = (1 / FastEnergy + c_fission * FastChi) * f_ssunshielded
# f_NRIMthermal = c_thermalNRIM * ThermalEnergy / kT * np.exp(-ThermalEnergy / kT)
# f_NRIMfast = (1 / FastEnergy + c_fission * FastChi) * f_ssNRIMFast
# f_NRthermal = c_thermalNR * ThermalEnergy / kT * np.exp(-ThermalEnergy / kT)
# f_NRfast = (1 / FastEnergy + c_fission * FastChi) * f_ssNRFast
#
# # Plot and save
# plt.figure(1, figsize=(20, 10))
#
# # Clears the figure from anything that was on it:
# plt.clf()
#
# # Plot what you want to plot:
# plt.plot(ThermalEnergy, f_unshieldedthermal, color='r', linestyle='-', label='$f_{unshielded}(E)$')
# plt.plot(FastEnergy, f_unshieldedfast, color='r', linestyle='-')
# plt.plot(FastEnergy, f_unshieldedfast, color='m', linestyle='-', label='$f_{unshielded}(E)$ with T = 600 K')
# plt.plot(ThermalEnergy, f_NRIMthermal, color='b', linestyle='-', label='$f_{NRIM}(E)$')
# plt.plot(FastEnergy, f_NRIMfast, color='b', linestyle='-')
# plt.plot(ThermalEnergy, f_NRthermal, color='g', linestyle='-', label='$f_{NR}(E)$')
# plt.plot(FastEnergy, f_NRfast, color='g', linestyle='-')
# plt.axvline(E_0, color='k', linestyle=':')
# plt.axvline(E_1, color='k', linestyle=':')
# plt.axvline(E_2, color='k', linestyle=':')
# plt.axvline(E_3, color='k', linestyle=':')
# plt.axvline(E_4, color='k', linestyle=':')
# plt.axvline(E_5, color='k', linestyle=':')
# plt.axvline(E_6, color='k', linestyle=':')
#
# # Label the axes:
# plt.legend(loc='best')
# plt.xlabel('Energy (eV)')
# plt.ylabel('Neutron Flux $\phi_{g}(E)$ $(n/{(cm^{3}*s*MeV)})$')
# plt.title('Neutron Flux in Three Regions')
#
# # Change the limits of the plot:
# plt.xlim([1e-5, 20e6])
#
# # Make the x variable logarithmically spaced:
# plt.xscale('log')
# plt.yscale('log')
#
# ############################################################ (#7)
#
# Group1STUS = (integrate(E_1, E_0, E, ST * f_unshieldedfast) / integrate(E_1, E_0, E, f_unshieldedfast))
# Group2STUS = (integrate(E_2, E_1, E, ST * f_unshieldedfast) / integrate(E_2, E_1, E, f_unshieldedfast))
# Group3STUS = (integrate(E_3, E_2, E, ST * f_unshieldedfast) / integrate(E_3, E_2, E, f_unshieldedfast))
# Group4STUS = (integrate(E_4, E_3, E, ST * f_unshieldedfast) / integrate(E_4, E_3, E, f_unshieldedfast))
# Group5STUS = (integrate(E_5, E_4, E, ST * f_unshieldedfast) / integrate(E_5, E_4, E, f_unshieldedfast))
# Group6STUS = (integrate(E_6, E_5, E, ST * f_unshieldedthermal) / integrate(E_6, E_5, E, f_unshieldedthermal))
# Group1STNRIM = (integrate(E_1, E_0, E, ST * f_NRIMfast) / integrate(E_1, E_0, E, f_NRIMfast))
# Group2STNRIM = (integrate(E_2, E_1, E, ST * f_NRIMfast) / integrate(E_2, E_1, E, f_NRIMfast))
# Group3STNRIM = (integrate(E_3, E_2, E, ST * f_NRIMfast) / integrate(E_3, E_2, E, f_NRIMfast))
# Group4STNRIM = (integrate(E_4, E_3, E, ST * f_NRIMfast) / integrate(E_4, E_3, E, f_NRIMfast))
# Group5STNRIM = (integrate(E_5, E_4, E, ST * f_NRIMfast) / integrate(E_5, E_4, E, f_NRIMfast))
# Group6STNRIM = (integrate(E_6, E_5, E, ST * f_NRIMthermal) / integrate(E_6, E_5, E, f_NRIMthermal))
# Group1STNR = (integrate(E_1, E_0, E, ST * f_NRfast) / integrate(E_1, E_0, E, f_NRfast))
# Group2STNR = (integrate(E_2, E_1, E, ST * f_NRfast) / integrate(E_2, E_1, E, f_NRfast))
# Group3STNR = (integrate(E_3, E_2, E, ST * f_NRfast) / integrate(E_3, E_2, E, f_NRfast))
# Group4STNR = (integrate(E_4, E_3, E, ST * f_NRfast) / integrate(E_4, E_3, E, f_NRfast))
# Group5STNR = (integrate(E_5, E_4, E, ST * f_NRfast) / integrate(E_5, E_4, E, f_NRfast))
# Group6STNR = (integrate(E_6, E_5, E, ST * f_NRthermal) / integrate(E_6, E_5, E, f_NRthermal))
# Group1SAUS = (integrate(E_1, E_0, E, SA * f_unshieldedfast) / integrate(E_1, E_0, E, f_unshieldedfast))
# Group2SAUS = (integrate(E_2, E_1, E, SA * f_unshieldedfast) / integrate(E_2, E_1, E, f_unshieldedfast))
# Group3SAUS = (integrate(E_3, E_2, E, SA * f_unshieldedfast) / integrate(E_3, E_2, E, f_unshieldedfast))
# Group4SAUS = (integrate(E_4, E_3, E, SA * f_unshieldedfast) / integrate(E_4, E_3, E, f_unshieldedfast))
# Group5SAUS = (integrate(E_5, E_4, E, SA * f_unshieldedfast) / integrate(E_5, E_4, E, f_unshieldedfast))
# Group6SAUS = (integrate(E_6, E_5, E, SA * f_unshieldedthermal) / integrate(E_6, E_5, E, f_unshieldedthermal))
# Group1SANRIM = (integrate(E_1, E_0, E, SA * f_NRIMfast) / integrate(E_1, E_0, E, f_NRIMfast))
# Group2SANRIM = (integrate(E_2, E_1, E, SA * f_NRIMfast) / integrate(E_2, E_1, E, f_NRIMfast))
# Group3SANRIM = (integrate(E_3, E_2, E, SA * f_NRIMfast) / integrate(E_3, E_2, E, f_NRIMfast))
# Group4SANRIM = (integrate(E_4, E_3, E, SA * f_NRIMfast) / integrate(E_4, E_3, E, f_NRIMfast))
# Group5SANRIM = (integrate(E_5, E_4, E, SA * f_NRIMfast) / integrate(E_5, E_4, E, f_NRIMfast))
# Group6SANRIM = (integrate(E_6, E_5, E, SA * f_NRIMthermal) / integrate(E_6, E_5, E, f_NRIMthermal))
# Group1SANR = (integrate(E_1, E_0, E, SA * f_NRfast) / integrate(E_1, E_0, E, f_NRfast))
# Group2SANR = (integrate(E_2, E_1, E, SA * f_NRfast) / integrate(E_2, E_1, E, f_NRfast))
# Group3SANR = (integrate(E_3, E_2, E, SA * f_NRfast) / integrate(E_3, E_2, E, f_NRfast))
# Group4SANR = (integrate(E_4, E_3, E, SA * f_NRfast) / integrate(E_4, E_3, E, f_NRfast))
# Group5SANR = (integrate(E_5, E_4, E, SA * f_NRfast) / integrate(E_5, E_4, E, f_NRfast))
# Group6SANR = (integrate(E_6, E_5, E, SA * f_NRthermal) / integrate(E_6, E_5, E, f_NRthermal))
# Group1SNFUS = (integrate(E_1, E_0, E, SNF * f_unshieldedfast) / integrate(E_1, E_0, E, f_unshieldedfast))
# Group2SNFUS = (integrate(E_2, E_1, E, SNF * f_unshieldedfast) / integrate(E_2, E_1, E, f_unshieldedfast))
# Group3SNFUS = (integrate(E_3, E_2, E, SNF * f_unshieldedfast) / integrate(E_3, E_2, E, f_unshieldedfast))
# Group4SNFUS = (integrate(E_4, E_3, E, SNF * f_unshieldedfast) / integrate(E_4, E_3, E, f_unshieldedfast))
# Group5SNFUS = (integrate(E_5, E_4, E, SNF * f_unshieldedfast) / integrate(E_5, E_4, E, f_unshieldedfast))
# Group6SNFUS = (integrate(E_6, E_5, E, SNF * f_unshieldedthermal) / integrate(E_6, E_5, E, f_unshieldedthermal))
# Group1SNFNRIM = (integrate(E_1, E_0, E, SNF * f_NRIMfast) / integrate(E_1, E_0, E, f_NRIMfast))
# Group2SNFNRIM = (integrate(E_2, E_1, E, SNF * f_NRIMfast) / integrate(E_2, E_1, E, f_NRIMfast))
# Group3SNFNRIM = (integrate(E_3, E_2, E, SNF * f_NRIMfast) / integrate(E_3, E_2, E, f_NRIMfast))
# Group4SNFNRIM = (integrate(E_4, E_3, E, SNF * f_NRIMfast) / integrate(E_4, E_3, E, f_NRIMfast))
# Group5SNFNRIM = (integrate(E_5, E_4, E, SNF * f_NRIMfast) / integrate(E_5, E_4, E, f_NRIMfast))
# Group6SNFNRIM = (integrate(E_6, E_5, E, SNF * f_NRIMthermal) / integrate(E_6, E_5, E, f_NRIMthermal))
# Group1SNFNR = (integrate(E_1, E_0, E, SNF * f_NRfast) / integrate(E_1, E_0, E, f_NRfast))
# Group2SNFNR = (integrate(E_2, E_1, E, SNF * f_NRfast) / integrate(E_2, E_1, E, f_NRfast))
# Group3SNFNR = (integrate(E_3, E_2, E, SNF * f_NRfast) / integrate(E_3, E_2, E, f_NRfast))
# Group4SNFNR = (integrate(E_4, E_3, E, SNF * f_NRfast) / integrate(E_4, E_3, E, f_NRfast))
# Group5SNFNR = (integrate(E_5, E_4, E, SNF * f_NRfast) / integrate(E_5, E_4, E, f_NRfast))
# Group6SNFNR = (integrate(E_6, E_5, E, SNF * f_NRthermal) / integrate(E_6, E_5, E, f_NRthermal))
# print('All XS are in 1/cm')
# print()
# Sigma_Total_Table = [['G1:', Group1STUS, Group1STNRIM, Group1STNR],
#                      ['G2:', Group2STUS, Group2STNRIM, Group2STNR],
#                      ['G3:', Group3STUS, Group3STNRIM, Group3STNR],
#                      ['G4:', Group4STUS, Group4STNRIM, Group4STNR],
#                      ['G5:', Group5STUS, Group5STNRIM, Group5STNR],
#                      ['G6:', Group6STUS, Group6STNRIM, Group6STNR]]
# print()
# print(tabulate(Sigma_Total_Table, headers=['', 'Total US XS', 'Total NRIM XS', 'Total NR XS'], tablefmt='orgtbl'))
#
# Sigma_Absorption_Table = [['G1:', Group1SAUS, Group1SANRIM, Group1SANR],
#                           ['G2:', Group2SAUS, Group2SANRIM, Group2SANR],
#                           ['G3:', Group3SAUS, Group3SANRIM, Group3SANR],
#                           ['G4:', Group4SAUS, Group4SANRIM, Group4SANR],
#                           ['G5:', Group5SAUS, Group5SANRIM, Group5SANR],
#                           ['G6:', Group6SAUS, Group6SANRIM, Group6SANR]]
# print()
# print(tabulate(Sigma_Absorption_Table, headers=['', 'Abs. US XS', 'Abs. NRIM XS', 'Abs. NR XS'], tablefmt='orgtbl'))
#
# Sigma_NuFission_Table = [['G1:', Group1SNFUS, Group1SNFNRIM, Group1SNFNR],
#                          ['G2:', Group2SNFUS, Group2SNFNRIM, Group2SNFNR],
#                          ['G3:', Group3SNFUS, Group3SNFNRIM, Group3SNFNR],
#                          ['G4:', Group4SNFUS, Group4SNFNRIM, Group4SNFNR],
#                          ['G5:', Group5SNFUS, Group5SNFNRIM, Group5SNFNR],
#                          ['G6:', Group6SNFUS, Group6SNFNRIM, Group6SNFNR]]
# print()
# print(tabulate(Sigma_NuFission_Table, headers=['', 'Fis. US XS', 'Fis. NRIM XS', 'Fis. NR XS'], tablefmt='orgtbl'))
#
# ############################################################ (#8)
#
# SigmaEnergy = np.array([E_6, E_5, E_4, E_3, E_2, E_1, E_0])
#
# # Come up with some MG XS
# sigmaMGUS = np.array([Group6SAUS, Group5SAUS, Group4SAUS,
#                       Group3SAUS, Group2SAUS, Group1SAUS])
# sigmaMGNRIM = np.array([Group6SANRIM, Group5SANRIM, Group4SANRIM,
#                         Group3SANRIM, Group2SANRIM, Group1SANRIM])
# sigmaMGNR = np.array([Group6SANR, Group5SANR, Group4SANR,
#                       Group3SANR, Group2SANR, Group1SANR])
#
# # Plot the XS
# plt.figure(figsize=(20, 10))
# plt.clf()
# plt.loglog(E, SA, color='dimgray', label='$\Sigma_\gamma(E)$', alpha=0.7)
#
# # Plot the MG XS as a histogram
# x, y = get_histogram(SigmaEnergy, sigmaMGUS)
# plt.loglog(x, y, label='$\Sigma_{unshielded \gamma,g}$', linewidth=2)
# x, y = get_histogram(SigmaEnergy, sigmaMGNRIM)
# plt.loglog(x, y, label='$\Sigma_{NRIM \gamma,g}$', linewidth=2)
# x, y = get_histogram(SigmaEnergy, sigmaMGNR)
# plt.loglog(x, y, label='$\Sigma_{NR \gamma,g}$', linewidth=2)
#
# # Plot the group boundaries
# plt.axvline(E_0, color='k', linestyle=':')
# plt.axvline(E_1, color='k', linestyle=':')
# plt.axvline(E_2, color='k', linestyle=':')
# plt.axvline(E_3, color='k', linestyle=':')
# plt.axvline(E_4, color='k', linestyle=':')
# plt.axvline(E_5, color='k', linestyle=':')
# plt.axvline(E_6, color='k', linestyle=':')
# plt.xlabel('Energy (eV)')
# plt.ylabel('Microscopic Cross Section (b)')
# plt.title('Absorption XS for different groups')
# plt.xlim([1e-5, 20e6])
# plt.legend(loc='best')
# plt.show()
#
# ############################################################ (#9)
#
# Sigma_Total_Table = [['G1:', abs(1 - Group1STUS / Group1STNR), abs(1 - Group1STNRIM / Group1STNR)],
#                      ['G2:', abs(1 - Group2STUS / Group2STNR), abs(1 - Group2STNRIM / Group2STNR)],
#                      ['G3:', abs(1 - Group3STUS / Group3STNR), abs(1 - Group3STNRIM / Group3STNR)],
#                      ['G4:', abs(1 - Group4STUS / Group4STNR), abs(1 - Group4STNRIM / Group4STNR)],
#                      ['G5:', abs(1 - Group5STUS / Group5STNR), abs(1 - Group5STNRIM / Group5STNR)],
#                      ['G6:', abs(1 - Group6STUS / Group6STNR), abs(1 - Group6STNRIM / Group6STNR)]]
# print()
# print(tabulate(Sigma_Total_Table, headers=['', 'US/NR Error', 'NRIM/NR Error'], tablefmt='orgtbl'))
#
# Sigma_Absorption_Table = [['G1:', abs(1 - Group1SAUS / Group1SANR), abs(1 - Group1SANRIM / Group1SANR)],
#                           ['G2:', abs(1 - Group2SAUS / Group2SANR), abs(1 - Group2SANRIM / Group2SANR)],
#                           ['G3:', abs(1 - Group3SAUS / Group3SANR), abs(1 - Group3SANRIM / Group3SANR)],
#                           ['G4:', abs(1 - Group4SAUS / Group4SANR), abs(1 - Group4SANRIM / Group4SANR)],
#                           ['G5:', abs(1 - Group5SAUS / Group5SANR), abs(1 - Group5SANRIM / Group5SANR)],
#                           ['G6:', abs(1 - Group6SAUS / Group6SANR), abs(1 - Group6SANRIM / Group6SANR)]]
# print()
# print(tabulate(Sigma_Absorption_Table, headers=['', 'US/NR Error', 'NRIM/NR Error'], tablefmt='orgtbl'))
#
# Sigma_NuFission_Table = [['G1:', abs(1 - Group1SNFUS / Group1SNFNR), abs(1 - Group1SNFNRIM / Group1SNFNR)],
#                          ['G2:', abs(1 - Group2SNFUS / Group2SNFNR), abs(1 - Group2SNFNRIM / Group2SNFNR)],
#                          ['G3:', abs(1 - Group3SNFUS / Group3SNFNR), abs(1 - Group3SNFNRIM / Group3SNFNR)],
#                          ['G4:', abs(1 - Group4SNFUS / Group4SNFNR), abs(1 - Group4SNFNRIM / Group4SNFNR)],
#                          ['G5:', abs(1 - Group5SNFUS / Group5SNFNR), abs(1 - Group5SNFNRIM / Group5SNFNR)],
#                          ['G6:', abs(1 - Group6SNFUS / Group6SNFNR), abs(1 - Group6SNFNRIM / Group6SNFNR)]]
# print()
# print(tabulate(Sigma_NuFission_Table, headers=['', 'US/NR Error', 'NRIM/NR Error'], tablefmt='orgtbl'))
# print()
#
# ############################################################ (#10)
#
# T = 600  # K
# k = 8.6173324 * 10 ** -5  # eV/K
# c_thermalunshielded = (((1 / E_thermal + c_fission * UnionChiU235[np.where(E == E_thermal)]) * f_ssunshielded) / (
# E_thermal / (k * T) * np.exp(-E_thermal / (k * T))))
#
# f_unshieldedthermal = c_thermalunshielded * ThermalEnergy / (k * T) * np.exp(-ThermalEnergy / (k * T))
# f_unshieldedfast = (1 / FastEnergy + c_fission * FastChi) * f_ssunshielded
#
# # 1/cm
# Group6STUS = (integrate(E_6, E_5, E, ST * f_unshieldedthermal) / integrate(E_6, E_5, E, f_unshieldedthermal))
# Group6SAUS = (integrate(E_6, E_5, E, SA * f_unshieldedthermal) / integrate(E_6, E_5, E, f_unshieldedthermal))
# Group6SNFUS = (integrate(E_6, E_5, E, SNF * f_unshieldedthermal) / integrate(E_6, E_5, E, f_unshieldedthermal))
#
# plt.figure(1)
# plt.plot(ThermalEnergy, f_unshieldedthermal, color='m')
# plt.plot(FastEnergy, f_unshieldedfast, color='m')
#
# print('Group 6 Total Unshielded XS (1/cm):', Group6STUS)
# print('Group 6 Absorption Unshielded XS (1/cm):', Group6SAUS)
# print('Group 6 NuFission Unshielded XS (1/cm):', Group6SNFUS)
# print()
#
# # Don't worry too much about this
# # It basically allows this file to be run from the command line
# # or to be loaded as a library
# # if __name__ == '__main__':
# #     run_problem()