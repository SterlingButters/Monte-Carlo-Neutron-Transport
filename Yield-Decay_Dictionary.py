import numpy as np
import random
import re

Dictionary = {
        "239-Pu": {"name": "239-Pu", "half-life": 2.411e4*3.154e7, "decay mode": "alpha", "mass": 239},
        "235-U" : {"name": "235-U",  "half-life": 7.038e8*3.154e+7, "decay mode": "alpha", "mass": 235},
        "231-Th": {"name": "231-Th", "half-life": 25.52*24*60**2, "decay mode": "beta", "mass": 231},
        "231-Pa": {"name": "231-Pa", "half-life": 3.2760e4*3.154e+7, "decay mode": "alpha", "mass": 231},
        "227-Ac": {"name": "227-Ac", "half-life": 21.773*3.154e+7, "decay mode": "beta", "mass": 227},
        "227-Th": {"name": "227-Th", "half-life": 18.72*24*60**2, "decay mode": "alpha", "mass": 227},
        "223-Ra": {"name": "223-Ra", "half-life": 11.435*24*60**2, "decay mode": "alpha", "mass": 223},
        "219-Rn": {"name": "219-Rn", "half-life": 3.96, "decay mode": "alpha", "mass": 219},
        "215-Po": {"name": "215-Po", "half-life": 1.781e-3, "decay mode": "alpha", "mass": 215},
        "211-Pb": {"name": "211-Pb", "half-life": 36.1*60, "decay mode": "beta", "mass": 211},
        "211-Bi": {"name": "211-Bi", "half-life": 2.14*60, "decay mode": "alpha", "mass": 211},
        "207-Tl": {"name": "207-Tl", "half-life": 4.77*60, "decay mode": "beta", "mass": 207},
        "207-Pb": {"name": "207-Pb", "half-life": 2.411e4, "decay mode": "stable", "mass": 207},
}
#
# # Loop dictionary for desired decay mode: alpha
# for entry in Dictionary:
#     if Dictionary[entry]["decay mode"] == "alpha":
#         # Print elements decay mode
#         print(Dictionary[entry]["name"], "decays by alpha emission")
#
# # Loop dictionary for desired decay mode: beta
# for entry in Dictionary:
#     if Dictionary[entry]["decay mode"] == "beta":
#         result = 6.022e23/Dictionary[entry]["mass"]*np.log(2)/Dictionary[entry]["half-life"]
#         # Print activity based on half-life
#         print("The activity of 1 gram for", Dictionary[entry]["name"], "beta emitter is", result)

# Only added fission products with > 1% yield; might need to added those with < 1% yield if they have significant
# cross-section... yields expressed in %


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            return item


######################################################################################################

y_vec_1 = []
Atomic_num_vec = []
file_1 = "/Users/juliadevinney/PycharmProjects/MonteCarlo/Independent Fission Yields/2D Plot Data for U-235 " \
       "Neutron-induced Fission Yields at 0.0253 (eV).txt"
with open(file_1, 'r+') as f:
    lines = f.readlines()
    for i in range(len(lines) - 4):
        line = lines[i + 4]
        nums = (re.findall("\S+", line))
        Atomic_num_vec.append(int(nums[0]))
        y_vec_1.append(float(nums[1]))

y_vec_1 = y_vec_1 / np.sum(y_vec_1)
pick1 = random_pick(Atomic_num_vec, y_vec_1)

######################################################################################################

y_vec_2 = []
Mass_vec = []
file_2 = "/Users/juliadevinney/PycharmProjects/MonteCarlo/Independent Fission Yields/A 2D Plot Data for U-235 " \
       "Neutron-induced Fission Yields at 0.0253 (eV).txt"
with open(file_2, 'r+') as f:
    lines = f.readlines()
    for i in range(len(lines) - 4):
        line = lines[i + 4]
        nums = (re.findall("\S+", line))
        Mass_vec.append(int(nums[0]))
        y_vec_2.append(float(nums[1]))

y_vec_2 = y_vec_2 / np.sum(y_vec_2)

######################################################################################################
