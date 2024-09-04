from typing import Any
import numpy as np
from shipsim import EssoOsaka_xu, Wind, InukaiPond, ManeuveringSimulation
from numba import jit
from my_src import cma_log2csv 

class Collision_Sim:
    def __init__(self, x_hat0):
        self.ship = EssoOsaka_xu()
        self.wind = Wind(mode="stationary")
        self.world = InukaiPond(wind=self.wind)
        self.sim = ManeuveringSimulation(
            ship=self.ship,
            world=self.world,
            dt_act=1.0,
            dt_sim=0.1,
            solve_method="rk4",
            log_dir="./log/sim/",
            check_collide=False,
        )
        self.state = x_hat0[0:]
        self.u = self.state[1]
        self.action = np.array([0, -20])
        self.x_hat = self.state[0:6]
        
    def __call__(self):
        self.sim.reset(self.state, seed=100)
        # print(self.state)
        # print(self.u)
        
    @jit    
    def emg_stop(self):
        # for i in range(10):
        # x_hat_list = []
        self.x_hat_array = np.full((10, 6), np.nan)
        self.x_hat_array[0] = self.x_hat
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
            
        # print("stop")
        # print("x_hat_array", self.x_hat_array)
        return self.x_hat_array
    
    def stop_distance(self):
        u_velo_list = []
        v_m_list = []
        stop_distance_list = []
        x_lists = {f"x{i}_list": [] for i in range(10)}
        self.x_hat_array = np.full((10, 6), np.nan)
        self.x_hat_array[0] = self.x_hat
        u = np.arange(0, 0.2, 0.001)
        v = np.arange(-0.04, 0.04, 0.01)
        for j, v_m in enumerate(v):
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
                for i in range(10):
                    x_lists[f"x{i}_list"].append(self.x_hat_array[i, 0].copy())

                        
        print(u.shape)
        # print(x_array.shape)
        # u_stop_dict = {"u_velo":u_velo_list, "v_m":v_m_list, "stop_distance":stop_distance_list}
        u_stop_dict = {
                        "u_velo":u_velo_list, 
                        "v_m":v_m_list,
                        **{f"x{i}":x_lists[f"x{i}_list"] for i in range(10)}
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
    
            