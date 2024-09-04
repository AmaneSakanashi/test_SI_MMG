from typing import Any
import numpy as np
import shipsim
from numba import jit
from my_src import cma_log2csv 
from utils.font import font_setting

class Collision_Sim:
    def __init__(self, x_hat0):
        self.ship = shipsim.EssoOsaka_xu()
        self.wind = shipsim.Wind(mode="stationary")
        self.world = shipsim.InukaiPond(wind=self.wind)
        self.sim = shipsim.ManeuveringSimulation(
            ship=self.ship,
            world=self.world,
            dt_act=1.0,
            dt_sim=0.1,
            solve_method="rk4", # "euler" or "rk4"
            log_dir="./log/sim_data/",
            check_collide=False,
        )
        self.state = x_hat0[0:]
        self.u = self.state[1]
        self.x_hat = self.state[0:6]
        self.action = np.array([0, -20])
        
    def __call__(self):
        self.sim.reset(self.state, seed=100)
        # print(self.state)
        # print(self.u)
        
    @jit    
    def emg_stop_full(self, count):
        # for i in range(10):
        # x_hat_list = []
        self.action = np.array([0, -20])
        self.x_hat_array = np.full((100, 6), np.nan)
        self.x_hat_array[0] = self.x_hat
        for i in range(1, 100): 
            if self.u < 0:
                break
            self.sim.get_t()
            observation, terminated, info = self.sim.step(self.action)
            # obs = observation[0:6]
            # print(obs)
            self.x_hat = info["state"][0:6]
            # shape [i, 6]
            # x_hat_list.append(self.x_hat)
            self.x_hat_array[i] = (self.x_hat)
            # t_list.append(sim.get_t())
            self.u = self.x_hat[1]
            # print("self.u", self.u)
    
        # font_setting()

        # self.sim.log2csv('emg__stop' + f'{count}')
        # self.sim.log2img('emg_stop' + f'{count}', ext_type='pdf')
            
        # print("stop")
        # print("x_hat_array", self.x_hat_array)
        return self.x_hat_array
    
    @jit    
    def emg_stop_half(self, count):
        # for i in range(10):
        # x_hat_list = []
        self.action = np.array([0, -10])
        self.x_hat_array = np.full((100, 6), np.nan)
        self.x_hat_array[0] = self.x_hat
        for i in range(1, 100): 
            if self.u < 0:
                break
            self.sim.get_t()
            observation, terminated, info = self.sim.step(self.action)
            # obs = observation[0:6]
            # print(obs)
            self.x_hat = info["state"][0:6]
            # shape [i, 6]
            # x_hat_list.append(self.x_hat)
            self.x_hat_array[i] = (self.x_hat)
            # t_list.append(sim.get_t())
            self.u = self.x_hat[1]
            # print("self.u", self.u)
    
        # font_setting()

        # self.sim.log2csv('emg__stop' + f'{count}')
        # self.sim.log2img('emg_stop' + f'{count}', ext_type='pdf')
            
        # print("stop")
        # print("x_hat_array", self.x_hat_array)
        return self.x_hat_array
    
    @jit    
    def emg_stop_rudder_stopped(self, count):
        # for i in range(10):
        # x_hat_list = []
        self.action = np.array([self.state[6], -20])
        self.x_hat_array = np.full((100, 6), np.nan)
        self.x_hat_array[0] = self.x_hat
        for i in range(1, 100): 
            if self.u < 0:
                break
            self.sim.get_t()
            observation, terminated, info = self.sim.step(self.action)
            # obs = observation[0:6]
            # print(obs)
            self.x_hat = info["state"][0:6]
            # shape [i, 6]
            # x_hat_list.append(self.x_hat)
            self.x_hat_array[i] = (self.x_hat)
            # t_list.append(sim.get_t())
            self.u = self.x_hat[1]
            # print("self.u", self.u)
        # font_setting()

        # self.sim.log2csv('emg__stop' + f'{count}')
        # self.sim.log2img('emg_stop' + f'{count}', ext_type='pdf')
            
        # print("stop")
        # print("x_hat_array", self.x_hat_array)
        return self.x_hat_array
    
    @jit    
    def tol_culc(self, count):
        self.action = np.array([0, -20])
        self.x_hat_array = np.full((100, 6), np.nan)
        tol_max = np.zeros(6)
        tol_min = np.zeros(6)
        for i in range(6):
            col_flag_plus = None
            col_flag_minus = None
            for j in range(100):
                if col_flag_plus and col_flag_minus:
                    break
                tol = (j+1) * 0.1 * self.x_hat[i]
                self.x_hat_array = self.x_hat[i] + tol
                for j in range(1, 100): 
                    if col_flag_plus:
                        break
                    if self.u < 0:
                        break
                    self.sim.get_t()
                    observation, terminated, info = self.sim.step(self.action)
                    self.x_hat = info["state"][0:6]
                    self.x_hat_array[j] = (self.x_hat)
                    self.u = self.x_hat[1]
                    col_flag_plus = col_check()
                
                if col_flag_plus and tol_max[i] == 0:
                    tol_max[i] = tol
                
                self.x_hat_array = self.x_hat[i] - tol
                for j in range(1, 100): 
                    if col_flag_minus:                        
                        break
                    if self.u < 0:
                        break
                    self.sim.get_t()
                    observation, terminated, info = self.sim.step(self.action)
                    self.x_hat = info["state"][0:6]
                    self.x_hat_array[j] = (self.x_hat)
                    self.u = self.x_hat[1]
                    col_flag_minus = col_check()
                    
                
                if col_flag_minus and tol_min[i] == 0:
                    tol_min[i] = tol
                    
        return self.x_hat_array, tol_max, tol_min
    
    @jit    
    def tol_culc(self, count):
        self.action = np.array([0, -20])
        self.x_hat_array = np.full((100, 6), np.nan)
        tol_max = np.zeros(6)
        tol_min = np.zeros(6)
        for i in range(6):
            tol = (j+1) * 0.1 * self.x_hat[i]
            self.x_hat_array = self.x_hat[i] + tol
            for j in range(1, 100):
                if col_flag_plus:
                    break
                if self.u < 0:
                    break
                self.sim.get_t()
                observation, terminated, info = self.sim.step(self.action)
                self.x_hat = info["state"][0:6]
                self.x_hat_array[j] = (self.x_hat)
                self.u = self.x_hat[1]
                col_flag_plus = col_check()
            
            if col_flag_plus and tol_max[i] == 0:
                tol_max[i] = tol
            
            self.x_hat_array = self.x_hat[i] - tol
            for j in range(1, 100): 
                if col_flag_minus:                        
                    break
                if self.u < 0:
                    break
                self.sim.get_t()
                observation, terminated, info = self.sim.step(self.action)
                self.x_hat = info["state"][0:6]
                self.x_hat_array[j] = (self.x_hat)
                self.u = self.x_hat[1]
                col_flag_minus = col_check()
                
            
            if col_flag_minus and tol_min[i] == 0:
                tol_min[i] = tol
                    
        return self.x_hat_array, tol_max, tol_min
    
    
    def stop_distance(self):
        u_velo_list = []
        v_m_list = []
        stop_distance_list = []
        x_list = []
        self.x_hat_array = np.full((30, 6), np.nan)
        self.x_hat_array[0] = self.x_hat
        u = np.arange(0, 0.2, 0.001)
        v = np.arange(-0.04, 0.04, 0.01)
        for v_m, j in v, range(len(v)):
            for u_velo in u:
                self.u = u_velo
                self.state[1] = self.u
                self.state[3] = v_m
                self.sim.reset(self.state, seed=100)
                for i in range(1, 10): 
                    if self.u < 0:
                        break
                    self.sim.get_t()
                    observation, terminated, info = self.sim.step(self.action)
                    # obs = observation[0:6]
                    # print(obs)
                    self.x_hat = info["state"][0:6]
                    # shape [i, 6]
                    # x_hat_list.append(self.x_hat)
                    self.x_hat_array[i] = (self.x_hat)
                    # t_list.append(sim.get_t())
                    self.u = self.x_hat[1]
                    # print("self.u", self.u)
                    # print(stop_distance)
                    
                u_velo_list.append(u_velo.copy())
                v_m_list.append(v_m.copy())
                stop_distance_list.append(self.x_hat[0].copy())
                x_list[j].append(self.x_hat_array[:, 0].copy())
                    
        
        # u_stop_dict = {"u_velo":u_velo_list, "v_m":v_m_list, "stop_distance":stop_distance_list}
        u_stop_dict = {
                        "u_velo":u_velo_list, 
                        "x0":x_list[0],
                        "x1":x_list[1],
                        "x2":x_list[2],
                        "x3":x_list[3],
                        "x4":x_list[4],
                        "x5":x_list[5],
                        "x6":x_list[6],
                        "x7":x_list[7],
                        "x8":x_list[8],
                        "x9":x_list[9],
        } 
        print(u_stop_dict)
        
        # # f = "stop_distance.txt"
        # path_w = 'test_w.txt'
        # with open(path_w, mode='w') as f:
        #     for key, value in u_stop_dict.items():
        #         f.write(f'{key}:{value}\n')
        #         # f.write(u_stop_dict)
        #     # f.write("aaaaaa")
        
        cma_log2csv(u_stop_dict, log_dir = "./my_src/log/", filename="stop_distance_2")
        
        
            



if __name__ == "__main__":
    L = 3
    B = 0.48925
    x_position_mid = 0
    u_velo = 0
    y_position_mid = 0
    vm_velo =  0
    psi = np.pi
    r_angvelo = 0.0

    delta_rudder = 0
    n_prop = 0

    true_wind_speed = 0
    true_wind_direction = np.pi
    
    x_hat0 = np.array([
    x_position_mid, u_velo, y_position_mid, vm_velo, psi, r_angvelo,
    delta_rudder, n_prop,
    true_wind_speed, true_wind_direction
    ])
    coll_sim = Collision_Sim(x_hat0)
    coll_sim()
    coll_sim.stop_distance()
    print("Task complited")
    
            