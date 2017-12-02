from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly as py
import plotly.tools as tls
import Cross_Section_Loading
import numpy as np
import pandas as pd
init_notebook_mode(connected=True)

# Wishlist (In order): Reflector, Control Rods, Source Rods, Full-Core, Concurrency


def energy_lookup(data_set, inp_energy):
    """look up energy in a data set and return the nearest energy in the table
    Input:
        data_set: a vector of energies
        inp_energy: the energy to lookup
    Output:
        index: the index of the nearest neighbor in the table
    """
    # argmin returns the indices of the smallest members of an array
    # here we’ll look for the minimum difference between the input energy and the table
    index = np.argmin(np.fabs(np.array(data_set)-inp_energy))
    return index

##################################################################

FUEL_PIN_RADIUS = .5
HEIGHT = 100
PITCH = 3

# assert(PITCH >= 2 * FUEL_PIN_RADIUS)

ENRICHMENT_1 = .03
ENRICHMENT_2 = .05
ENRICHMENT_3 = .07

# ASSEMBLY = [[1, 1, 1, 1, 1, 1, 1, 3, 3],  # Quarter Section
#             [1, 1, 1, 1, 1, 1, 1, 3, 3],
#             [1, 1, 2, 2, 2, 2, 2, 3, 3],
#             [1, 1, 2, 1, 1, 1, 2, 3, 3],
#             [1, 1, 2, 1, 1, 1, 2, 3, 3],
#             [1, 1, 2, 1, 1, 1, 2, 3, 3],
#             [1, 1, 2, 2, 2, 2, 2, 3, 3],
#             [3, 3, 3, 3, 3, 3, 3, 3, 3],
#             [3, 3, 3, 3, 3, 3, 3, 3, 3]]

# Leakage Test
ASSEMBLY = [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # Quarter Section
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]


##################################################################

print("For 17 x 17 assembly, 9 x 9 is needed. Your shape:", np.shape(ASSEMBLY))

X_DIM = np.shape(ASSEMBLY)[1]
Y_DIM = np.shape(ASSEMBLY)[0]

X_BOUNDARY = X_DIM * PITCH
Y_BOUNDARY = Y_DIM * PITCH

centers_x = []
centers_y = []
centers_z = []
fuel_type = []

data = []

for i in range(X_DIM):

    for j in range(Y_DIM):

        if ASSEMBLY[j][i] != 0:

            u = np.linspace(0,  2*np.pi, 50)

            # Cylinders (#'s denote quadrant)
            x_1 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.cos(u)) + PITCH * i
            y_1 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.sin(u)) + PITCH * j
            z_1 = np.outer(np.linspace(0, HEIGHT, np.size(u)), np.ones(np.size(u)))

            x_2 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.cos(u)) + PITCH * -i
            y_2 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.sin(u)) + PITCH * j
            z_2 = np.outer(np.linspace(0, HEIGHT, np.size(u)), np.ones(np.size(u)))

            x_3 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.cos(u)) + PITCH * i
            y_3 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.sin(u)) + PITCH * -j
            z_3 = np.outer(np.linspace(0, HEIGHT, np.size(u)), np.ones(np.size(u)))

            x_4 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.cos(u)) + PITCH * -i
            y_4 = FUEL_PIN_RADIUS * np.outer(np.ones(np.size(u)), np.sin(u)) + PITCH * -j
            z_4 = np.outer(np.linspace(0, HEIGHT, np.size(u)), np.ones(np.size(u)))

            # ['Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 'Greens', 'Greys', 'Hot', 'Jet', 'Picnic', 'Portland', 'Rainbow', 'RdBu', 'Reds', 'Viridis', 'YlGnBu', 'YlOrRd']

            lateral_surface_1 = go.Surface(x=x_1,
                                           y=y_1,
                                           z=z_1,
                                           colorscale='Greys',
                                           surfacecolor=np.ones(np.size(u)),
                                           opacity=.25,
                                           )

            lateral_surface_2 = go.Surface(x=x_2,
                                           y=y_2,
                                           z=z_2,
                                           colorscale='Greys',
                                           surfacecolor=np.ones(np.size(u)),
                                           opacity=.25,
                                           )

            lateral_surface_3 = go.Surface(x=x_3,
                                           y=y_3,
                                           z=z_3,
                                           colorscale='Greys',
                                           surfacecolor=np.ones(np.size(u)),
                                           opacity=.25,
                                           )

            lateral_surface_4 = go.Surface(x=x_4,
                                           y=y_4,
                                           z=z_4,
                                           colorscale='Greys',
                                           surfacecolor=np.ones(np.size(u)),
                                           opacity=.25,
                                           )

            centers_x.append(PITCH * i)
            centers_y.append(PITCH * j)
            centers_z.append(HEIGHT)
            fuel_type.append(ASSEMBLY[j][i])

            centers_x.append(PITCH * -i)
            centers_y.append(PITCH * j)
            centers_z.append(HEIGHT)
            fuel_type.append(ASSEMBLY[j][i])

            centers_x.append(PITCH * i)
            centers_y.append(PITCH * -j)
            centers_z.append(HEIGHT)
            fuel_type.append(ASSEMBLY[j][i])

            centers_x.append(PITCH * -i)
            centers_y.append(PITCH * -j)
            centers_z.append(HEIGHT)
            fuel_type.append(ASSEMBLY[j][i])

            data.append(lateral_surface_1)
            data.append(lateral_surface_2)
            data.append(lateral_surface_3)
            data.append(lateral_surface_4)

centers = go.Scatter3d(x=centers_x,
                       y=centers_y,
                       z=centers_z,
                       mode='markers',
                       marker=dict(
                           size=5,
                           ))
                      

# data.append(centers)

##################################################################
# Remove Duplicates
assert (len(centers_x) == len(centers_y))
pre_dup_list = []
rem_dup_list = []
for a in range(len(centers_x)):
    pre_dup_list.append((centers_x[a], centers_y[a]))

for tup in pre_dup_list:
    if tup not in rem_dup_list:
        rem_dup_list.append(tup)

centers_x = []
centers_y = []
for tup in rem_dup_list:
    centers_x.append(tup[0])
    centers_y.append(tup[1])

##################################################################


def material_type(x, y):
    assert(len(centers_x) == len(centers_y))
    for a in range(len(centers_x)):
        # Define domain of r for each pin center
        r_domain = np.sqrt((x - centers_x[a])**2 + (y - centers_y[a])**2)
        # See if x and y are in fuel pin
        if r_domain <= FUEL_PIN_RADIUS:
            return fuel_type[a]

    return False

##################################################################
# Load Cross-Sections

E = Cross_Section_Loading.E
# Uranium 238 Cross-sections
U238_scattering = Cross_Section_Loading.UnionElasticU238
U238_absorption = Cross_Section_Loading.UnionGammaU238
U238_fission = Cross_Section_Loading.UnionFissionU238
U238_gamma = Cross_Section_Loading.UnionGammaU238
U238_total = Cross_Section_Loading.UnionTotalU238
U238_nu = Cross_Section_Loading.UnionNuU238

# Uranium 235 Cross-sections
U235_scattering = Cross_Section_Loading.UnionElasticU235
U235_absorption = Cross_Section_Loading.UnionGammaU235
U235_fission = Cross_Section_Loading.UnionFissionU235
U235_gamma = Cross_Section_Loading.UnionGammaU235
U235_total = Cross_Section_Loading.UnionTotalU235
U235_nu = Cross_Section_Loading.UnionNuU235

# Hydrogen & Oxygen Cross-Sections
hydrogen_scattering = Cross_Section_Loading.UnionElasticH1
hydrogen_absorption = Cross_Section_Loading.UnionGammaH1
hydrogen_gamma = Cross_Section_Loading.UnionGammaH1
hydrogen_total = Cross_Section_Loading.UnionTotalH1

oxygen_scattering = Cross_Section_Loading.UnionElasticO16
oxygen_absorption = Cross_Section_Loading.UnionGammaO16
oxygen_gamma = Cross_Section_Loading.UnionGammaO16
oxygen_total = Cross_Section_Loading.UnionTotalO16

##################################################################

A_fuel = 238             # Needs Effective Atomic Number?
DENSITY_FUEL = 10.97

A_mod = 7.42             # Effective Atomic Number
DENSITY_MOD = 1
N_mod = DENSITY_MOD / A_mod * 6.022e23 * 1e-24
N_H_mod = N_mod * 2      # Number of atoms per molecule (2)
N_O_mod = N_mod * 1      # Number of atoms per molecule (1)

alpha_fuel = (A_fuel - 1.0) ** 2 / (A_fuel + 1.0) ** 2
alpha_mod = (A_mod - 1.0) ** 2 / (A_mod + 1.0) ** 2

# Fuel Macroscopic Cross-Sections
def sigma_total_fuel(enrich):
    N_U235 = enrich * DENSITY_FUEL * (6.022e23 * 1e-24)/235.0439
    N_U238 = (1 - enrich) * DENSITY_FUEL * (6.022e23 * 1e-24)/238.0508
    N_O16 = (N_U235 + N_U238) * 2
    return N_U235 * U235_total + N_U238 * U238_total + N_O16 * oxygen_total


def sigma_scatter_fuel(enrich):
    N_U235 = enrich * DENSITY_FUEL * (6.022e23 * 1e-24)/235.0439
    N_U238 = (1 - enrich) * DENSITY_FUEL * (6.022e23 * 1e-24)/238.0508
    N_O16 = (N_U235 + N_U238) * 2
    return N_U235 * U235_scattering + N_U238 * U238_scattering + N_O16 * oxygen_scattering


def sigma_capture_fuel(enrich):
    N_U235 = enrich * DENSITY_FUEL * (6.022e23 * 1e-24)/235.0439
    N_U238 = (1 - enrich) * DENSITY_FUEL * (6.022e23 * 1e-24)/238.0508
    N_O16 = (N_U235 + N_U238) * 2
    return N_U235 * U235_gamma + N_U238 * U238_gamma + N_O16 * oxygen_gamma


def sigma_absorb_fuel(enrich):
    N_U235 = enrich * DENSITY_FUEL * (6.022e23 * 1e-24)/235.0439
    N_U238 = (1 - enrich) * DENSITY_FUEL * (6.022e23 * 1e-24)/238.0508
    N_O16 = (N_U235 + N_U238) * 2
    return N_U235 * U235_absorption + N_U238 * U238_absorption + N_O16 * oxygen_absorption


def sigma_fission_fuel(enrich):
    N_U235 = enrich * DENSITY_FUEL * (6.022e23 * 1e-24)/235.0439         
    N_U238 = (1 - enrich) * DENSITY_FUEL * (6.022e23 * 1e-24)/238.0508   
    return N_U235 * U235_fission + N_U238 * U238_fission


def nu_fuel(enrich):
    return enrich * U235_nu + (1 - enrich) * U238_nu

# Moderator Macroscopic Cross-Sections
sigma_total_mod = N_O_mod * oxygen_total + N_H_mod * hydrogen_total
sigma_scatter_mod = N_O_mod * oxygen_scattering + N_H_mod * hydrogen_scattering
sigma_capture_mod = N_O_mod * oxygen_gamma + N_H_mod * hydrogen_gamma

##################################################################

# get distance to interface from inside fuel pin
# Come up with alternative to Ridder function EDIT HERE


def distance_to_edge(x, y, phi):

    intersect_list = []
    for a in range(len(centers_x)):
        # Construct Line Equation
        m = np.tan(phi)
        c = y - x * m

        # Line-Circle Intersection is Quadratic
        A = m ** 2 + 1
        B = m * c - m * centers_x[a] - centers_y[a]
        C = centers_y[a] ** 2 - FUEL_PIN_RADIUS ** 2 + centers_x[a] ** 2 - 2 * c * centers_y[a] + c ** 2

        # Determine number of intersections and roots
        # if B**2 - 4*A*C < 0:
        #     print("No Intersection")
        #     print(rem_dup_list[a])
        #
        # if B**2 - 4*A*C == 0:
        #     print("One Intersection")
        #     print(rem_dup_list[a])

        # In this case there are two intersections
        if B**2 - 4*A*C > 0:
            # print("Two Intersections")

            # Two quadratic results (x)
            x_intersect1 = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            x_intersect2 = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)

            # # Two quadratic results (y)
            y_intersect1 = x_intersect1*np.tan(phi) + c
            y_intersect2 = x_intersect2*np.tan(phi) + c

            tup1 = (x_intersect1, y_intersect1)
            tup2 = (x_intersect2, y_intersect2)

            # If neutron is going "up" in the x-y plane of the assembly
            if phi < np.pi:
                if y_intersect1 > y:
                    if not np.isnan(tup1[0]):
                        intersect_list.append(tup1)

                if y_intersect2 > y:
                    if not np.isnan(tup2[0]):
                        intersect_list.append(tup2)

            # If neutron is going "down" in the x-y plane of the assembly
            else:
                if y_intersect1 < y:
                    if not np.isnan(tup1[0]):
                        intersect_list.append(tup1)

                if y_intersect2 < y:
                    if not np.isnan(tup2[0]):
                        intersect_list.append(tup2)

    d_vec = []
    for a in range(len(intersect_list)):
        # Find distance from neutron origin to circle intersection
        d = np.sqrt((x - intersect_list[a][0]) ** 2 + (y - intersect_list[a][1]) ** 2)
        d_vec.append(float(d))

    # Take closest distance
    if len(d_vec) > 0:
        # print(np.min(d_vec))
        return np.min(d_vec)

    if len(d_vec) == 0:
        # should I return something else?
        return 'no-interface'

##################################################################


def particle_func(x, y, z):
    nu = 0

    death_eng = 0  
    death_loc = 0  
    death_type = 0 

    # get initial direction
    theta = np.random.uniform(0, np.pi, 1)
    phi = np.random.uniform(0, 2 * np.pi, 1)

    # compute energy via rejection sampling
    expfiss = lambda e: 0.453 * np.exp(-1.036 * e / 1.0e6) * np.sinh(np.sqrt(2.29 * e / 1.0e6))

    min_eng = np.min(E)
    max_eng = np.max(E)
    max_prob = np.max(expfiss(E))

    thermal_boundary = 0.0253

    rejected = 1
    while rejected:
        a = np.random.uniform(min_eng, max_eng, 1)
        b = np.random.uniform(0, max_prob, 1)
        rel_prob = expfiss(a)
        if b <= rel_prob:
            energy = a
            rejected = 0

    alive = 1

    # vector to keep track of positions
    xvec = np.ones(1) * x
    yvec = np.ones(1) * y
    zvec = np.ones(1) * z

    while alive:
        # Get real/new cross-sections for corresponding energy
        index = energy_lookup(E, energy)

        interacted = 0
        total_distance = 0

        # Interacted may still be alive (scattering)
        while interacted == 0:

            ###################################################
            # Determine starting location for sample distance using sigma_total
            material_start = material_type(x, y)

            if material_start == 1:
                sig_tot = sigma_total_fuel(ENRICHMENT_1)[index]
            elif material_start == 2:
                sig_tot = sigma_total_fuel(ENRICHMENT_2)[index]
            elif material_start == 3:
                sig_tot = sigma_total_fuel(ENRICHMENT_3)[index]
            else:
                sig_tot = sigma_total_mod[index]

            ###################################################

            if material_start == 1 or material_start == 2 or material_start == 3:  # if in fuel pin

                # Get distance to edge of fuel rod (from fuel)
                d = distance_to_edge(x, y, phi)

                # get sample distance to collision
                s = -np.log(1.0 - np.random.random(1)) / sig_tot

                # Incidence on interface (denoted by code "no-interface")
                if d != 'no-interface':

                    # Sample distance is greater than interface distance (does not account for material change)
                    # Must convert between 2D and 3D
                    if s * np.sin(theta) > d:
                        total_distance += d / np.sin(theta)

                    # Sample distance is correct and interaction occurs
                    else:
                        total_distance += s
                        interacted = 1

                # Statement may be redundant but idk how to handle return from distance_to_rod
                else:
                    total_distance += s
                    interacted = 1

            else:               # if in moderator
                # get distance to edge of fuel rod (from moderator)
                d = distance_to_edge(x, y, phi)

                # get distance to collision
                s = -np.log(1.0 - np.random.random(1)) / sig_tot

                # Incidence on interface (denoted by code "no-interface")
                if d != 'no-interface':

                    # Sample distance is greater than interface distance (does not account for material change)
                    # Must convert between 2D and 3D
                    if s * np.sin(theta) > d:
                        total_distance += d / np.sin(theta)  # <- Right conversion?

                    # Sample distance is correct and interaction occurs
                    else:
                        total_distance += s
                        interacted = 1

                # Statement may be redundant but idk how to handle return from distance_to_rod
                else:
                    total_distance += s
                    interacted = 1

            # move particle
            z += total_distance * np.cos(theta)
            y += total_distance * np.sin(theta) * np.sin(phi)
            x += total_distance * np.sin(theta) * np.cos(phi)

        # material_end = material_type(x, y)
        #
        # if material_start != material_end:
        #     print("Neutron has crossed material interface(s)")

        # Trace/Track particle movement
        xvec = np.append(xvec, x)
        yvec = np.append(yvec, y)
        zvec = np.append(zvec, z)

        ###################################################

        # Leakage
        if x > X_BOUNDARY or x < -X_BOUNDARY or y > Y_BOUNDARY or y < -Y_BOUNDARY or z > HEIGHT or z < 0:
            alive = 0

            death_type = 'leak'

            if energy < thermal_boundary:
                death_eng = 'thermal'
            else:
                death_eng = 'fast'

        ###################################################

        # Determine Type of interaction based on energy and corresponding cross-sections
        # In fuel
        material = material_type(x, y)
        if material == 1:
            sig_scat_temp = sigma_scatter_fuel(ENRICHMENT_1)[index]
            sig_abs_temp = sigma_absorb_fuel(ENRICHMENT_1)[index]
            sig_fiss_temp = sigma_fission_fuel(ENRICHMENT_1)[index]
            sig_tot_temp = sigma_total_fuel(ENRICHMENT_1)[index]
            nu_temp = nu_fuel(ENRICHMENT_1)[index]

            # Scatter
            if np.random.random(1) < sig_scat_temp / sig_tot_temp:

                # scatter, pick new angles & energy
                theta = np.random.uniform(0, np.pi, 1)
                phi = np.random.uniform(0, 2 * np.pi, 1)
                energy = np.random.uniform(alpha_fuel * energy, energy, 1)

            elif np.random.random(1) < sig_abs_temp / sig_tot_temp:

                # Fission
                if np.random.random(1) < sig_fiss_temp / sig_abs_temp:

                    # Determine number of neutrons produced from fission (round/int?)
                    nu = int(round(nu_temp))
                    alive = 0
                    death_type = 'fission'

                # No Fission
                else:
                    alive = 0
                    death_type = 'absorption'

                death_loc = 'fuel'

                if energy < thermal_boundary:
                    death_eng = 'thermal'
                else:
                    death_eng = 'fast'

        #############################

        elif material == 2:
            sig_scat_temp = sigma_scatter_fuel(ENRICHMENT_2)[index]
            sig_abs_temp = sigma_absorb_fuel(ENRICHMENT_2)[index]
            sig_fiss_temp = sigma_fission_fuel(ENRICHMENT_2)[index]
            sig_tot_temp = sigma_total_fuel(ENRICHMENT_2)[index]
            nu_temp = nu_fuel(ENRICHMENT_2)[index]

            # Scatter
            if np.random.random(1) < sig_scat_temp / sig_tot_temp:

                # scatter, pick new angles & energy
                theta = np.random.uniform(0, np.pi, 1)
                phi = np.random.uniform(0, 2 * np.pi, 1)
                energy = np.random.uniform(alpha_fuel * energy, energy, 1)

            elif np.random.random(1) < sig_abs_temp / sig_tot_temp:

                # Fission
                if np.random.random(1) < sig_fiss_temp / sig_abs_temp:

                    # Determine number of neutrons produced from fission (round/int?)
                    nu = int(round(nu_temp))
                    alive = 0
                    death_type = 'fission'

                # No Fission
                else:
                    alive = 0
                    death_type = 'absorption'

                death_loc = 'fuel'

                if energy < thermal_boundary:
                    death_eng = 'thermal'
                else:
                    death_eng = 'fast'

        #############################

        if material == 3:
            sig_scat_temp = sigma_scatter_fuel(ENRICHMENT_3)[index]
            sig_abs_temp = sigma_absorb_fuel(ENRICHMENT_3)[index]
            sig_fiss_temp = sigma_fission_fuel(ENRICHMENT_3)[index]
            sig_tot_temp = sigma_total_fuel(ENRICHMENT_3)[index]
            nu_temp = nu_fuel(ENRICHMENT_3)[index]

            # Scatter
            if np.random.random(1) < sig_scat_temp / sig_tot_temp:

                # scatter, pick new angles & energy
                theta = np.random.uniform(0, np.pi, 1)
                phi = np.random.uniform(0, 2 * np.pi, 1)
                energy = np.random.uniform(alpha_fuel * energy, energy, 1)

            elif np.random.random(1) < sig_abs_temp / sig_tot_temp:

                # Fission
                if np.random.random(1) < sig_fiss_temp / sig_abs_temp:

                    # Determine number of neutrons produced from fission (round/int?)
                    nu = int(round(nu_temp))
                    alive = 0
                    death_type = 'fission'

                # No Fission
                else:
                    alive = 0
                    death_type = 'absorption'

                death_loc = 'fuel'

                if energy < thermal_boundary:
                    death_eng = 'thermal'
                else:
                    death_eng = 'fast'

        #############################

        # In water
        else:
            mod_scat = sigma_scatter_mod[index]
            mod_tot = sigma_total_mod[index]

            # Scatter
            if np.random.random(1) < mod_scat / mod_tot:

                # scatter, pick new angles & energy
                theta = np.random.uniform(0, np.pi, 1)
                phi = np.random.uniform(0, 2 * np.pi, 1)
                energy = np.random.uniform(alpha_mod * energy, energy, 1)

            else:
                # absorbed
                alive = 0
                death_type = 'absorption'

                death_loc = 'mod'

        ###################################################
    death_fate = [death_loc, death_eng, death_type]

    return xvec, yvec, zvec, nu, death_fate

##################################################################
gen_count = 0

thermal_leaked = 0
fast_leaked = 0

thermal_absorbed_fuel = 0
fast_absorbed_fuel = 0

thermal_absorbed_mod = 0
fast_absorbed_mod = 0

thermal_fission = 0
fast_fission = 0

thermal_produced = 0
fast_produced = 0
##########################

# for i in pyprind.prog_bar(range(N), track_time=True, monitor=True, bar_char='█', width=50,):
N = 500

for i in range(N):

    ###################################################

    # Uniformly Dispersed Source (Cylinder)
    # x = np.random.uniform(-X_BOUNDARY, X_BOUNDARY, 1)
    # y = np.random.uniform(-Y_BOUNDARY, Y_BOUNDARY, 1)
    # z = np.random.uniform(-HEIGHT, HEIGHT, 1)

    # Uniformly Dispersed FUEL Source (Cylinder)
    rejected = 1
    while rejected:
        x = np.random.uniform(-X_BOUNDARY, X_BOUNDARY, 1)
        y = np.random.uniform(-Y_BOUNDARY, Y_BOUNDARY, 1)
        z = np.random.uniform(-HEIGHT, HEIGHT, 1)
        if material_type(x, y):
            rejected = 0

    ###################################################

    # Get normal particle info (trace)
    x_vec, y_vec, z_vec, nu, fate = particle_func(x, y, z)

    ###############################################

    gen_count += nu

    if fate[2] == 'leak':
        if fate[0] == 'thermal':
            thermal_leaked += 1
        if fate[1] == 'fast':
            fast_leaked += 1

    if fate[0] == 'fuel' and fate[1] == 'thermal':
        thermal_absorbed_fuel += 1
        if fate[2] == 'fission':
            thermal_fission += 1
            thermal_produced += nu

    if fate[0] == 'fuel' and fate[1] == 'fast':
        fast_absorbed_fuel += 1
        if fate[2] == 'fission':
            fast_fission += 1
            fast_produced += nu

    if fate[0] == 'mod' and fate[1] == 'thermal':
        thermal_absorbed_mod += 1

    if fate[0] == 'mod' and fate[1] == 'fast':
        fast_absorbed_mod += 1

    ###############################################

    particle_trace = go.Scatter3d(
        x=x_vec,
        y=y_vec,
        z=z_vec,
        mode='lines',
        line=dict(color='rgb(173, 255, 47)')
    )

    data.append(particle_trace)

    ###################################################

    nu_vec = [nu]
    x_vecs = [x_vec]
    y_vecs = [y_vec]
    z_vecs = [z_vec]

    if nu > 0:
        print("{} neutrons generated for neutron {}".format(nu, i))

    else:
        print("No neutrons generated for neutron {}".format(i + 1))

    t = 0
    recent_nus = nu_vec
    while np.any(recent_nus) != 0:

        print(nu_vec[-t:])

        tracker = 0

        nu_vec_temp = []

        x_vecs_temp = []
        y_vecs_temp = []
        z_vecs_temp = []

        for a in range(len(nu_vec[-t:])):

            x = x_vecs[-(a + 1)][-1]
            y = y_vecs[-(a + 1)][-1]
            z = z_vecs[-(a + 1)][-1]

            for j in range(nu_vec[-(a + 1)]):
                x_vec, y_vec, z_vec, nu, fate = particle_func(x, y, z)

                ###############################################

                gen_count += nu

                if fate[2] == 'leak':
                    if fate[0] == 'thermal':
                        thermal_leaked += 1
                    if fate[1] == 'fast':
                        fast_leaked += 1

                if fate[0] == 'fuel' and fate[1] == 'thermal':
                    thermal_absorbed_fuel += 1
                    if fate[2] == 'fission':
                        thermal_fission += 1
                        thermal_produced += nu

                if fate[0] == 'fuel' and fate[1] == 'fast':
                    fast_absorbed_fuel += 1
                    if fate[2] == 'fission':
                        fast_fission += 1
                        fast_produced += nu

                if fate[0] == 'mod' and fate[1] == 'thermal':
                    thermal_absorbed_mod += 1

                if fate[0] == 'mod' and fate[1] == 'fast':
                    fast_absorbed_mod += 1

                ###############################################

                print("Particle {} starting coords:".format(j + 1), x_vec[0], y_vec[0], z_vec[0])
                print("Particle {} ending coords:".format(j + 1), x_vec[-1], y_vec[-1], z_vec[-1])
                print("Particle {} nu value".format(j + 1), nu)

                nu_vec_temp.append(nu)
                tracker += 1

                x_vecs_temp.append(x_vec)
                y_vecs_temp.append(y_vec)
                z_vecs_temp.append(z_vec)

                # time.sleep(1)

                particle_trace = go.Scatter3d(
                    x=x_vec,
                    y=y_vec,
                    z=z_vec,
                    mode='lines',
                    line=dict(color='rgb(255, 0, 0)')
                )

                data.append(particle_trace)

            print()
            t = tracker

        nu_vec.extend(nu_vec_temp)
        x_vecs.extend(x_vecs_temp)
        y_vecs.extend(y_vecs_temp)
        z_vecs.extend(z_vecs_temp)

        recent_nus = nu_vec_temp

        print("Continuing fission:", (np.any(recent_nus) != 0))

##################################################################

# Eta_F:
# Eta_T:
#   u_F:
#   f_T:
#     p:

Eta_T = thermal_produced / thermal_absorbed_fuel
Eta_F = fast_produced / fast_absorbed_fuel
u_F = fast_absorbed_fuel / (fast_absorbed_fuel + fast_absorbed_mod)
f_T = thermal_absorbed_fuel / (thermal_absorbed_fuel + fast_absorbed_fuel)
p = (thermal_absorbed_fuel + thermal_absorbed_mod + thermal_leaked) / (thermal_absorbed_fuel + fast_absorbed_fuel + thermal_absorbed_mod + fast_absorbed_mod + thermal_leaked)
epsilon_inf = 1 + (1-p) * u_F * Eta_F / (p * f_T * Eta_T)

k_inf = epsilon_inf * Eta_T * f_T * p

print("\nTotal number of neutrons left in system:", N + gen_count - (thermal_leaked + fast_leaked))
print("Total number of neutrons generated from {} neutron source: {}".format(N, gen_count))
print("Total number of leaked neutrons:", (thermal_leaked + fast_leaked))
print("Infinite Medium System Multiplication factor:", k_inf)
print("System Multiplication factor: (need 6 factor)", gen_count/N)

layout = go.Layout(
    title='Monte Carlo Assembly',
    autosize=True,
    showlegend=False,
    height=1000,
    width=1000,
    scene=dict(zaxis=dict(range=[-1, HEIGHT + 1]),
               yaxis=dict(range=[-(Y_DIM * PITCH + 5), (Y_DIM * PITCH + 5)]),
               xaxis=dict(range=[-(X_DIM * PITCH + 5), (X_DIM * PITCH + 5)])
               ),
)

fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename='file.html')
