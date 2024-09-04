import numpy as np
import pandas as pd
from numba import jit, prange
from multiprocessing import Pool
import logging
import ray
ray.init(num_cpus=32)

import os
import sys
import time
sys.path.append(os.getcwd())

import warnings
warnings.simplefilter('ignore')

import shipsim
import si_sim
from bound import *
from ddcma import *
from my_src import *
from guideline import *

from utils.font import font_setting

font_setting()

si = si_sim.SI_obj()

ship=shipsim.EssoOsaka_xu()

# Generation mode can be selected from "random" or "stationary".
wind = shipsim.Wind(mode="stationary")

world=shipsim.OpenSea(wind=wind)

sim = shipsim.ManeuveringSimulation(
    ship=ship,
    world=world,
    dt_act = 1.0,
    dt_sim=0.1,
    solve_method="rk4", # "euler" or "rk4"train_data
    log_dir="./log/sim_data/",
    check_collide=False,
)# you can access the name of state and action variavles
# print(f'State variables: {sim.STATE_NAME}')
# print(f'Action variables: {sim.ACTION_NAME}')

# # Define initial state variables––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
L = 3
B = 0.48925
x_position_mid = 20 * L
u_velo = 2.77 * 1852/3600
y_position_mid = -10 * B
vm_velo = 0.0
psi = np.pi 
r_angvelo = 0.0

delta_rudder = 0.0
n_prop = 20

true_wind_speed = 0
true_wind_direction = np.pi
### initial state ###
# init_state = np.array([
#     x_position_mid, u_velo, y_position_mid, vm_velo, psi, r_angvelo,
#     delta_rudder, n_prop,
#     true_wind_speed, true_wind_direction
# ])
# x_tol = np.array([0.01, 0.01, 0.01, 0.01, 0.005, 0.005]).T
# ### w_xxx : Weight of the Obj. term
# w_L = 0.1 * L
# w_U = 0.25 / 2
# w_dim = np.array([1/w_L**2, 1/w_U**2, 1/w_L**2, 1/w_U**2, np.pi**2, w_L**2/w_U**2]).T
# w_pen = 1e+3
# w_obj = np.array(x_tol,w_dim, w_pen)
# # -------------------------------------------------------------------------------------------------------------------------------------
# observation = sim.reset(init_state, seed=100)

# you can check the noise scale of the observation variavles
print(f'Normal noise scale: {sim.OBSERVATION_SCALE}')
### ===========================================================================-
### Setup for dd-CMA ###
# m = 21
# Setting for resart
NUM_RESTART = 8  # number of restarts with increased population size
MAX_NEVAL = 1e+8   # maximal number of f-calls
F_TARGET = 1e-8   # target function value
total_neval = 0   # total number of f-calls
iter_count = 0
### N: Dimention of CMA opt. target (maybe)
# N = 2*m+1
N = 31

# –--------------------------------------------------------------------------------------------------------------------------
print("N",N)
xmean0 = np.random.randn(N)
# print(len(xmean0))

lam_sig_mode = 0
if lam_sig_mode == 0:
    # 集団サイズ大，初期ステップサイズ大
    lam = 20 * N
    sigma0 = 3* np.ones(N)
elif lam_sig_mode == 1:
    # 集団サイズ大，初期ステップサイズ小
    lam = 20 * N
    sigma0 = 0.1 * np.ones(N)
elif lam_sig_mode == 2:
    # 集団サイズ小，初期ステップサイズ大
    lam = N
    sigma0 = 2 * np.ones(N)
else:
    # 集団サイズ小，初期ステップサイズ大
    lam = N
    sigma0 = 5 * np.ones(N) 

ddcma = DdCma(xmean0, sigma0, best_f=1e8)
# LOWER_BOUND, UPPER_BOUND, FLAG_PERIODIC = bound()
bound = set_bound()
LOWER_BOUND, UPPER_BOUND, FLAG_PERIODIC, period_length = bound.bound(N)

# ddcma.upper_bounding_coordinate_std(period_length)
checker = Checker(ddcma)
logger = Logger(ddcma)## ===============================================================================================-

## Initial setup so far ##
## ===============================================================================================

    

# Main loop
start = time.time()
time_diff = 0
for restart in range(NUM_RESTART):        
    issatisfied = False
    fbestsofar = np.inf
    while not issatisfied:
        # ddcma.onestep(func=fobj)
        ddcma.onestep(func=si.fobj)
        ddcma.upper_bounding_coordinate_std(period_length)
        fbest = np.min(ddcma.arf)
        fbestsofar = min(fbest, fbestsofar)

        if fbest <= F_TARGET:
            issatisfied, condition = True, 'ftarget'
        else:
            # checker.check_iqr()
            issatisfied, condition = checker()
        # if ddcma.t % 10 == 0:
            end = time.time()
            time_diff =  end - start
            print("===================================================")
            print("progress", ddcma.t, ddcma.neval, fbest, fbestsofar)
            # print("===================================================")
            #logpoint: "progress",{time_diff}, {ddcma.t}, {ddcma.neval}, {fbest} {fbestsofar}
            logger()
        # if ddcma.t % 100 == 0:
        #     # Produce a figure
        #     fig, axdict = logger.plot()
        #     for key in axdict:
        #         if key not in ('xmean'):
        #             axdict[key].set_yscale('log')
        #     plt.tight_layout()
        #     plt.savefig(logger.prefix + '.pdf')
    logger(condition)
    print("Terminated with condition: " + str(condition))
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # log_dir = f"./log/sim_data/"
    # prefix = f"test_{restart-1}"
    # csv_dir = f"{log_dir}{prefix}.csv"
    # df = pd.read_csv(csv_dir)
    # data_list = [df]
    # line_list = ["-"]
    # # print(x_hat_array.shape)
    
    # plot.state_plot(data_list=data_list, line_list=line_list, log_dir=log_dir, prefix=prefix)
    # plot.action_plot(data_list=data_list, line_list=line_list, log_dir=log_dir, prefix=prefix)
    # ----------------------------------------------------------------------------------------------------------------------------------------------    
    # For restart
    iter_count += 1
    total_neval += ddcma.neval
    if total_neval < MAX_NEVAL and fbest > F_TARGET:
        popsize = ddcma.lam * 2
        ddcma = DdCma(xmean0, sigma0, lam=popsize, best_f=10000)
        checker = Checker(ddcma)
        logger.setcma(ddcma)
        print("Restart with popsize: " + str(ddcma.lam))
    else:
        break

# Produce a figure
fig, axdict = logger.plot()
for key in axdict:
    if key not in ('xmean'):
        axdict[key].set_yscale('log')
plt.tight_layout()
plt.savefig(logger.prefix + '.pdf')
print("Task Completed")