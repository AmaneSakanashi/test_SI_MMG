import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import ray
from shipsim.ship import EssoOsaka_xu
from shipsim.world import OpenSea

from ddcma import *
from my_src import *



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
        
    
        # self.data = pd.read_csv("traindata/random_processed_Result_sim_0_06-Aug-2020_16_24_25_6.xlsx.csv",header=0, index_col=0 )
        # # time
        # self.startstep = 0 
        # self.no_timestep = len(self.data.loc[self.data.index[self.startstep:], 'x_position_mid [m]'].values)
       
        # self.action_train = np.empty((int(self.no_timestep), 2 ))
        # self.action_train[:,0]  = self.data.loc[self.data.index[self.startstep:], 'delta_rudder [rad]'].values
        # self.action_train[:,1]  = self.data.loc[self.data.index[self.startstep:], 'n_prop [rps]'].values
        
        # self.state_train = np.empty((int(self.no_timestep), 6 ))
        # self.state_train[:,0] = self.data.loc[self.data.index[self.startstep:], 'x_position_mid [m]'].values
        # self.state_train[:,1] = self.data.loc[self.data.index[self.startstep:], 'u_velo [m/s]'].values
        # self.state_train[:,2] = self.data.loc[self.data.index[self.startstep:], 'y_position_mid [m]'].values
        # self.state_train[:,3] = self.data.loc[self.data.index[self.startstep:], 'vm_velo [m/s]'].values
        # self.state_train[:,4] = self.data.loc[self.data.index[self.startstep:], 'psi_hat [rad]'].values
        # self.state_train[:,5] = self.data.loc[self.data.index[self.startstep:], 'r_angvelo [rad/s]'].values

        # self.wind_train = np.empty((int(self.no_timestep), 2 ))
        # self.wind_train[:,0] = self.data.loc[self.data.index[self.startstep:], 'wind_velo_true [m/s]'].values
        # self.wind_train[:,1] = self.data.loc[self.data.index[self.startstep:], 'wind_dir_true [rad]'].values
        self.no_files, self.no_timestep,\
        self.set_action_train, self.set_state_train, self.set_wind_train = Read_train.read_csv(self)

        self.sim = shipsim.ManeuveringSimulation(
            ship=ship,
            world=world,
            dt_act = 1.0,
            dt_sim = 0.1,
            solve_method="rk4", # "euler" or "rk4"train_data
            log_dir="./log/sim_data/",
            check_collide=False,
        )

        N = 62
        
        L = 3
        B = 0.48925
        # x_position_mid = 20 * L
        # u_velo = 2.77 * 1852/3600
        # y_position_mid = -10 * B
        # vm_velo = 0.0
        # psi = np.pi 
        # r_angvelo = 0.0
        # init_ship_state = self.state_train[0,:]
        # delta_rudder = 0.0
        # n_prop = 20
        # init_ship_action = self.action_train[0,:]

        # true_wind_speed = 0
        # true_wind_direction = np.pi
        # init_true_wind = self.wind_train[0,:]
        bound = Set_bound()   
        self.LOWER_BOUND, self.UPPER_BOUND, self.FLAG_PERIODIC, self.period_length = bound.set_param_bound(N)

        ### initial state ###
        # self.init_state = np.concatenate([init_ship_state,init_ship_action,init_true_wind])
        ### w_xxx : Weight of the Obj. term
        self.w_max = 1e+2
        self.w_noise = self.sim.ship.OBSERVATION_SCALE
        self.w_pen = 1e+8


    @ray.remote
    def trj_culc(self, x):
            w_noise, w_max, w_pen  = self.w_noise, self.w_max, self.w_pen
            update_params = x.copy()
            func = 0

            # t_list = []  
            for j in range(self.no_files):
                state_train = self.set_state_train[j,:,:]
                action_train = self.set_action_train[j,:,:]
                wind_train = self.set_wind_train[j,:,:]
                # state_train = cls.state_train
                # action_train = cls.action_train
                init_ship_state     = state_train[0,:]
                init_ship_action    = action_train[0,:]
                init_true_wind      = wind_train[0,:]

                init_state = np.concatenate([init_ship_state,init_ship_action,init_true_wind])

                self.sim.reset(init_state, seed=100)

                # ---------------------------------------------------------------------------------------------------------------
                    
                # ---------------------------------------------------------------------------------------------------------------
                ### calculating trj. loop ###
                for i in range(int(self.no_timestep)):    
                    t, state_sim = self.sim.step(update_params, action_train[i], wind_train[i])
                    # t_list.append(t)
                    #obj_func
                    func_i = Obj_function.J_3( state_sim, state_train[i], w_noise, w_max, w_pen,t )
                    func += func_i
                    # font_setting()
                actor = Obj_function()
                func_lkh = actor.J_lkh(update_params)
                func = func + func_lkh

            return func

class SI_obj:
    def __init__(self) -> None:
            self.const_Obj = 62 * (math.log( 2 * math.pi + 1 ))

    def fobj(self, x):
            actor = SI_trj()
            params_list = mirror(x, actor.LOWER_BOUND, actor.UPPER_BOUND, actor.FLAG_PERIODIC)
            cand_results = []
            
            for set_params in params_list:
                
                ## -------------------------------------------------------

                m_params = set_params[0:31]
                v_params = set_params[31:]

                ## For degug ---------------------------------------------
                # mean_result = pd.read_csv("log/cma/pattern1mean_result.csv", header=None)
                # var_result = pd.read_csv("log/cma/pattern1var_result.csv", header=None)
                # m_params = np.array(mean_result.values.flatten())
                # v_params = np.array(var_result.values.flatten())
                ## -------------------------------------------------------
                det_v_params = abs(np.sum(v_params))

                results = []
                gen_params = np.array([np.random.normal(m, v, 2) 
                        for m, v in zip(m_params,v_params)]).T
                for i in range(gen_params.ndim):
                    per_result_id = actor.trj_culc.remote(actor,gen_params[i]) 
                    # per_result_id = actor.trj_culc(gen_params[i]) 
                    results.append(per_result_id)

                results = ray.get(results)
                results_all = np.mean(results)   + \
                                0.5 *( math.log(det_v_params)+ self.const_Obj)
                cand_results.append(results_all)

            return  cand_results



