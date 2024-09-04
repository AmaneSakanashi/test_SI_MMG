import numpy as np
import time
from numba import jit

import shipsim
from ddcma import *
from my_src import J_2, mat_J_2, emg_stop_J_1, emg_stop_J_2
from my_src import collision_detection, mat_collision_detection
from my_src import calculate_ship_rectangle
from my_src import cma_log2csv
from my_src import obstacle, mat_obstacle
from my_src import Collision_Sim
from utils.font import font_setting

class SimulationParameters:
    def __init__(self, L, B, segment_duration, w_L, w_U, w_pen, x_fin, x_tol):
        self.L = L
        self.B = B
        self.segment_duration = segment_duration
        self.w_L = w_L
        self.w_U = w_U
        self.w_pen = w_pen
        self.x_fin = np.array([0.1 - self.L, 0, -0.1 + (-0.9) * self.B, 0, np.pi, 0]).T
        self.x_tol = np.array([0.01, 0.01, 0.01, 0.01, 0.005, 0.005]).T

def reset(self):
    self.ship=shipsim.EssoOsaka_xu()

    # Generation mode can be selected from "random" or "stationary".
    self.wind = shipsim.Wind(mode="stationary")

    self.world=shipsim.InukaiPond(wind=self.wind)

    self.sim = shipsim.ManeuveringSimulation(
        ship=self.ship,
        world=self.world,
        dt_act=1.0,
        dt_sim=1.0,
        solve_method="rk4", # "euler" or "rk4"
        log_dir="./log/sim_data/",
        check_collide=False,
    )
    L = 3
    B = 0.48925
    x_position_mid = 20 * L
    u_velo = 0.25
    y_position_mid = -5 * B
    vm_velo = 0.0
    psi = np.pi
    r_angvelo = 0.0

    delta_rudder = 0.0
    n_prop = 0

    true_wind_speed = 0
    true_wind_direction = np.pi

    self.state = np.array([
        x_position_mid, u_velo, y_position_mid, vm_velo, psi, r_angvelo,
        delta_rudder, n_prop,
        true_wind_speed, true_wind_direction
    ])
    
    self.poly_berth, self.poly_inukai = mat_obstacle()

@jit
def trj_culc(self, x, best_f, iter_count):
    self.sim.reset(self.state)
    # ---------------------------------------------------------------------------------------------------------------
    tf = x[0]
    x_hat_list = np.zeros((int(tf + 1), 6))
    t_list = []
    # collision_t_list = []
    current_segment = 0
    action = np.array([x[2], x[m+2]])
    x_stop_point_list = []
    x_hat = self.state[0:]
    # ---------------------------------------------------------------------------------------------------------------
    # print("tf:",tf,"[delta, n_prop]:", action)
    # print(sim.get_t())
    # ---------------------------------------------------------------------------------------------------------------
    start_time0 = time.time()
    for i in range(int(tf + 1)):
    # while sim.get_t() < tf:
        # print("sim_TIME:", sim.get_t())
        segment_start_time = current_segment * self.segment_duration
        if int(self.sim.get_t() + 0.1) >= segment_start_time + self.segment_duration:
            current_segment += 1
            action = np.array([x[current_segment + 2], x[current_segment + m + 2]])
            x_stop_point_list.append(x_hat.copy())
            # print(x_stop_point_list)
            # print("[delta, n_prop]:", action)
            
        observation, terminated, info = self.sim.step(action)
        x_hat = info["state"][0:]
        # print("x_hat:",x_hat)
        x_hat_list[i] = x_hat[0:6]
        t_list.append(self.sim.get_t())
    end_time0 = time.time()
    diff_time0 = end_time0 - start_time0
    print(f'diff_time0 : {diff_time0}')        
    func, error_array = J_3(tf, x_hat_list[-1], x_fin, x_tol, w_dim, w_pen)

    print("J_1:", func) 
    # ----------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    if func <= best_f-0.01:
        #メモリ削減する必要あり
        best_f = func
    
        font_setting()

        self.sim.log2csv('test' + f"_{iter_count}")
        self.sim.log2img(x_hat_array, 'test' + f"_{iter_count}", ext_type='pdf')
    # ----------------------------------------------------------------------------------------------------------------
    return func, best_f

# @jit
# def trj_culc(self, x, best_f, iter_count):
#     self.sim.reset(self.state)
#     # ---------------------------------------------------------------------------------------------------------------
#     tf = x[0]
#     x_hat_list = np.zeros((int(tf + 1), 6))
#     t_list = []
#     # collision_t_list = []
#     current_segment = 0
#     action = np.array([x[2], x[m+2]])
#     x_stop_point_list = []
#     x_hat = self.state[0:]
#     # ---------------------------------------------------------------------------------------------------------------
#     # print("tf:",tf,"[delta, n_prop]:", action)
#     # print(sim.get_t())
#     # ---------------------------------------------------------------------------------------------------------------
#     start_time0 = time.time()
#     for i in range(int(tf + 1)):
#     # while sim.get_t() < tf:
#         # print("sim_TIME:", sim.get_t())
#         segment_start_time = current_segment * self.segment_duration
#         if int(self.sim.get_t() + 0.1) >= segment_start_time + self.segment_duration:
#             current_segment += 1
#             action = np.array([x[current_segment + 2], x[current_segment + m + 2]])
#             x_stop_point_list.append(x_hat.copy())
#             # print(x_stop_point_list)
#             # print("[delta, n_prop]:", action)
            
#         observation, terminated, info = self.sim.step(action)
#         x_hat = info["state"][0:]
#         # print("x_hat:",x_hat)
#         x_hat_list[i] = x_hat[0:6]
#         t_list.append(self.sim.get_t())
#     end_time0 = time.time()
#     diff_time0 = end_time0 - start_time0
#     print(f'diff_time0 : {diff_time0}')        
#     # 緊急停止
#     # ver.1-----------------------------------------------------------------------------------------------------------------------
#     start_time1 = time.time()
#     x_hat_list0 = []  
#     for i in range(len(x_stop_point_list)):
#         # print(x_stop_point_list[i])
#         col_sim = Collision_Sim(x_stop_point_list[i])
#         col_sim()
#         emg_x = col_sim.emg_stop()
#         x_hat_list0.append(emg_x)
    
#     x_hat_array = np.array(x_hat_list0)
#     end_time1 = time.time()
#     diff_time1 = end_time1 - start_time1
#     print(f'diff_time:{diff_time1}')        
#     # print("x_hat_list0", x_hat_array.shape)
#     # print("x_hat_list0", x_hat_array)
#     # -------------------------------------------------------------------------------------------------------------------------------------------
    
#     # ver.2-----------------------------------------------------------------------------------------------------------------------------------
#     # start_time1 = time.time()
#     # x_stop_point_array = np.array(x_stop_point_list)
#     # data = pd.read_csv('my_src/log/stop_distance_2.csv') 
#     # x_hat_list0 = np.zeros((len(x_stop_point_list), 10, 6))
#     # x_hat_list0[:, :, :6] = np.transpose(np.tile(x_stop_point_array[:, :6, np.newaxis], (1, 1, 10)), (0, 2, 1))
#     # # print(f"x_hat:{x_hat_list0}")
    
#     # # for i in range(len(x_hat_list0)):
#     # #     for j in range(len(x_hat_list0[0])):
#     # #         init_u = x_hat_list0[i, j, 1]  
#     # #         for index, row in data.iterrows():
#     # #             if np.allclose(init_u, row['u_velo'], atol=1e-3):
#     # #                 move_distance = row[f'x{j}']
#     # #                 break
#     # #         x = x_hat_list0[i, j, 0]
#     # #         y = x_hat_list0[i, j, 2] 
#     # #         psi = x_hat_list0[i, j, 4]
#     # #         stopped_x = x + move_distance * np.cos(psi)
#     # #         stopped_y = y + move_distance * np.sin(psi)
#     # #         x_hat_list0[i, j, 0] = stopped_x
#     # #         x_hat_list0[i, j, 0] = stopped_y
    
#     # init_u_values = x_hat_list0[:, :, 1]
#     # matching_rows = data[data['u_velo'].values[:, np.newaxis] == init_u_values]
#     # move_distances = matching_rows.filter(regex='x\d').values
#     # print(move_distances)

#     # x, y, psi = x_hat_list0[:, :, [0, 2, 4]].T
#     # moved_x = move_distances * np.cos(psi)
#     # moved_y = move_distances * np.sin(psi)
#     # x_hat_list0[:, :, [0, 2]] += np.stack([moved_x, moved_y], axis=-1)

#     # # 最終的なx_hat_list0
#     # print(x_hat_list0)
 
#     # end_time1 = time.time()
#     # diff_time1 = end_time1 - start_time1
#     # print(f'diff_time1 : {diff_time1}')
            
#     # ------------------------------------------------------------------------------------------------------------------------------------------
    
#     # 船の形状判定-----------------------------------------------------------------------------------------------------
#     start_time2 = time.time()
#     p_array = np.zeros((len(x_hat_list0), 4, 2, 10))  # 4は頂点の数、2は座標の次元

#     for i in range(len(x_hat_list0)):
#         x_hat_list1 = np.array(x_hat_list0[i])
        
#         p_0, p_1, p_2, p_3 = calculate_ship_rectangle(x_hat_list1[:, 0], x_hat_list1[:, 2], x_hat_list1[:, 4], self.L, self.B)
#         # print("p_0",p_0)
        
#         p = np.array([p_0, p_1, p_2, p_3])
        
#         p_array[i] = p
        
#     end_time2 = time.time()
#     diff_time2 = end_time2 - start_time2
#     print(f'diff_time2:{diff_time2}')    
#     # ----------------------------------------------------------------------------------------------------------------

#     # 衝突判定------------------------------------------------------------------------------------------------------------
#     start_time3 = time.time()
#     for i in range(len(p_array)):
#         for j in range(len(p_array[0])):
#             indices =  mat_collision_detection(p_array[i][j], self.poly_berth, self.poly_inukai)
#             p_array[i][j][~np.isin(np.arange(len(p_array[i][j])), indices)] = np.nan
#     end_time3 = time.time()
#     diff_time3 = end_time3 - start_time3
#     print(f'diff_time3:{diff_time3}')    

#     #obj_func
#     # print(type(x_hat_list))
#     # print(type(t_list))
#     # print(type(p_array))

#     start_time4 = time.time() 
#     func = emg_stop_J_2(tf, x_hat_list[-1], self.x_fin, self.x_tol, p_array, self.w_dim, self.w_pen, w=10, y_berth=0.1)
#     end_time4 = time.time()
#     diff_time4 = end_time4 - start_time4
#     print(f'diff_time4:{diff_time4}')   
#     print("J_1:", func) 
#     # ----------------------------------------------------------------------------------------------------------------

#     # ----------------------------------------------------------------------------------------------------------------
#     if func <= best_f-0.01:
#         #メモリ削減する必要あり
#         best_f = func
    
#         font_setting()

#         self.sim.log2csv('test' + f"_{iter_count}")
#         self.sim.log2img(x_hat_array, 'test' + f"_{iter_count}", ext_type='pdf')
#     # ----------------------------------------------------------------------------------------------------------------
#     return func, best_f
        