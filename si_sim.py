import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import ray
from shipsim.ship import EssoOsaka_xu
from shipsim.world import OpenSea

from ddcma import *
from my_src import *
from guideline import *
from bound import *




class SI_trj:
    def __init__(
        self,
        ship=EssoOsaka_xu(),
        world=OpenSea(),
        dt_act=1.0,
        dt_sim=0.1,
    ):
        self.ship = ship
        self.world = world
        self.dt_act = dt_act
        self.dt_sim = dt_sim

        self.data = pd.read_csv("traindata/random_processed_Result_sim_0_06-Aug-2020_16_24_25_1.xlsx.csv",header=0, index_col=0 )
        # time
        self.startstep = 1200 
        self.no_timestep = len(self.data.loc[self.data.index[self.startstep:], 'x_position_mid [m]'].values)
       
        self.action_train = np.empty((int(self.no_timestep), 2 ))
        self.action_train[:,0]  = self.data.loc[self.data.index[self.startstep:], 'n_prop [rps]'].values
        self.action_train[:,1]  = self.data.loc[self.data.index[self.startstep:], 'delta_rudder [rad]'].values
        
        self.state_train = np.empty((int(self.no_timestep), 6 ))
        self.state_train[:,0] = self.data.loc[self.data.index[self.startstep:], 'x_position_mid [m]'].values
        self.state_train[:,1] = self.data.loc[self.data.index[self.startstep:], 'u_velo [m/s]'].values
        self.state_train[:,2] = self.data.loc[self.data.index[self.startstep:], 'y_position_mid [m]'].values
        self.state_train[:,3] = self.data.loc[self.data.index[self.startstep:], 'vm_velo [m/s]'].values
        self.state_train[:,4] = self.data.loc[self.data.index[self.startstep:], 'psi_hat [rad]'].values
        self.state_train[:,5] = self.data.loc[self.data.index[self.startstep:], 'r_angvelo [rad/s]'].values

        self.sim = shipsim.ManeuveringSimulation(
            ship=ship,
            world=world,
            dt_act = 1.0,
            dt_sim=0.1,
            solve_method="rk4", # "euler" or "rk4"train_data
            log_dir="./log/sim_data/",
            check_collide=False,
        )

        N = 31

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

        bound = set_bound()
        self.LOWER_BOUND, self.UPPER_BOUND, self.FLAG_PERIODIC, self.period_length = bound.bound(N)

        ### initial state ###
        self.init_state = np.array([
            x_position_mid, u_velo, y_position_mid, vm_velo, psi, r_angvelo,
            delta_rudder, n_prop,
            true_wind_speed, true_wind_direction
        ])
        self.x_tol = np.array([0.01, 0.01, 0.01, 0.01, 0.005, 0.005]).T
        ### w_xxx : Weight of the Obj. term
        w_L = 0.1 * L
        w_U = 0.25 / 2
        self.w_dim = np.array([1/w_L**2, 1/w_U**2, 1/w_L**2, 1/w_U**2, np.pi**2, w_L**2/w_U**2]).T
        self.w_pen = 1e+3

    # @ray.remote
    def trj_culc(self, x):
            self.sim.reset(self.init_state, seed=100)
            x_tol, w_dim, w_pen = self.x_tol, self.w_dim, self.w_pen 
            # x_tol, w_dim, w_pen = cls.x_tol, cls.w_dim, cls.w_pen 

            state_train = self.state_train
            action_train = self.action_train
            # state_train = cls.state_train
            # action_train = cls.action_train
            # ---------------------------------------------------------------------------------------------------------------
            update_params = x
            no_timestep = len(state_train)
            t_list = []        
            # ---------------------------------------------------------------------------------------------------------------
            ### calculating trj. loop ###
            for i in range(int(no_timestep)):    
                t, state_sim, observation_sim = self.sim.step(update_params, action_train[i])
                t_list.append(self.sim.get_t())
            
            #obj_func
                func_i, error_array = J_3( state_sim, state_train[i], x_tol, w_dim, w_pen)
                func =+ func_i

                font_setting()

            sim_log = self.sim.logger.log_pr()
            cma_log = [ error_array, x ]

            return func, sim_log, cma_log

class SI_obj:
    def fobj(self, x, f ):
            actor = SI_trj()
            xx_list = mirror(x, actor.LOWER_BOUND, actor.UPPER_BOUND, actor.FLAG_PERIODIC)

            # self.sim.reset(self.init_state, seed=100)
            # state_train = self.state_train
            # action_train = self.action_train
            # # ---------------------------------------------------------------------------------------------------------------
            # update_params = x
            # no_timestep = len(state_train)
            # t_list = []
            # f_list = []
            
            # # ---------------------------------------------------------------------------------------------------------------
            # ### calculating trj. loop ###
            # for i in range(int(no_timestep)):    
            #     t, state_sim, observation_sim = self.sim.step(update_params, action_train[i])
            #     t_list.append(sim.get_t())
            
            # #obj_func
            #     func_i, error_array = J_3( state_sim[i], state_train[i], x_tol, w_dim, w_pen)
            #     func =+ func_i

            #     font_setting()

            # sim_log = sim.logger.log_pr()
            # cma_log = [ error_array, x ]
            
            #-------------------------------------------------------------------------------------------------------------------------
            # result_ids = []
            # result_ids = [self.trj_culc(xx) for xx in xx_list]
            # for xx in xx_list:
            #     result_ids_xx = actor.trj_culc.remote(actor,xx)
            #     result_ids.append(result_ids_xx)

            results = [actor.trj_culc(xx) for xx in xx_list]
            # results = ray.get(result_ids)

            func_list, sim_log, cma_log = zip(*results)
            sorted_lists = sorted(zip(func_list, sim_log, cma_log))
            
            best_func, best_sim_log, best_cma_log = sorted_lists[0]
            error_array = best_cma_log[0]
            x = best_cma_log[1]

            df_log = pd.DataFrame(best_sim_log)
            df_log.to_csv(f"./log/sim_data/test.csv",  index=False, 
                        header=["t [s]","x_position_mid [m]","u_velo [m/s]","y_position_mid [m]","vm_velo [m/s]","psi [rad]","r_angvelo [rad/s]","delta_rudder [rad]","n_prop [rps]","true_wind_speed [m/s]","true_wind_direction [rad]"])
            # df_log.to_csv(f"./log/sim_data/simlation_trj_log.csv",  index=False)

            with open('./log/sim_data/DDCMA_log.txt', 'a') as f:
                f.write(f' f: {best_func}, \n error_array: {np.array2string(error_array)},\n x: {np.array2string(x)}\n')
                # f.write(f'ddcma.t: {ddcma.t}, ddcma.neval: {ddcma.neval}, f: {best_func}, \n error_array: {np.array2string(error_array)},\n x: {np.array2string(x)}\n')
            # if ddcma.t % 1000 == 1:
            #     log_dir = f"./log/sim_data/"
            #     prefix = f"test_{restart}"
            #     csv_dir = f"{log_dir}{prefix}.csv"
            #     df = pd.read_csv(csv_dir)
            #     data_list = [df]
            #     line_list = ["-"]
                
            #     plot.state_plot(data_list=data_list, line_list=line_list, log_dir=log_dir, prefix=prefix)
            #     plot.action_plot(data_list=data_list, line_list=line_list, log_dir=log_dir, prefix=prefix)
            #     # sim.viewer.get_x0y0_plot_2(prefix=prefix, csv_dir=csv_dir, p_array=None, x_emg_stop=x_hat_array[0], ext_type="pdf")
            # self.sim.reset(state=self.data_shokiti)

            return  func_list, best_func



    # si = SI()