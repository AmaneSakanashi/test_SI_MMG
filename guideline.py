import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d, splrep, splev, UnivariateSpline
from scipy.optimize import curve_fit

import shipsim
from my_src import Collision_Sim
from my_src import mat_collision_detection
from my_src import mat_obstacle
from my_src import speed_reducation, speed_reducation_guideline
from utils.font import font_setting
# from my_src.kaiseki import test
from my_src import plot

font_setting()

def plot_guideline(data_list, log_dir):
    #----------------------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.grid()
    for data in data_list:
        # print(data)
        # for index, row in data.iterrows():
        
        g = 9.8
        L = 3
        L_150 = 150
        B = 0.48925
        esso_V_N = 2.77 #(knot)
        x = data['x_position_mid [m]']
        y = data['y_position_mid [m]']
        u = data['u_velo [m/s]']
        psi = data['psi [rad]']
        # print(x, y ,u)
        x_fin = np.array([0.1-L, 0, -0.1+(-0.9)*B, 0, np.pi, 0]).T
        x_berth = x_fin[0]   
        y_berth = x_fin[1]
        berth_distance_array = np.sqrt((x - x_berth)**2 + (y - y_berth)**2)
        berth_distance_array_hat = berth_distance_array / L
        # berth_distance_array = berth_distance_array * 3600/1852
        knot_u = u * 3600/1852
        knot_u_150 = knot_u * np.sqrt(L_150/L)
        # esso_speed_hat = knot_u / esso_V_N 
        esso_speed_hat = u / np.sqrt(g * L)
        # print(berth_distance_array)
        ax.plot(berth_distance_array_hat, esso_speed_hat)
    #----------------------------------------------------------------------------------------------------------------------------------
    sensitivity_coef = 0.109
    constant_integration = 0.15
    
    ship_flag = 1
    if ship_flag == 0:
        V_N = 15.21 #knot 
        Lpp = 325
        berth_distance = np.linspace(20*Lpp, 0) 
        
        displacement =  324372 #ton
        # displacement = 200000
        added_mass = displacement * 0.1
        # added_mass = 200000 * 2625 / 37500
        brake_force = 14.1
        brake_per = brake_force / (displacement + added_mass)
        print(brake_per)
    elif ship_flag == 1:
        V_N = 14
        Lpp = 200
        berth_distance = np.linspace(20*Lpp, 0)
        displacement =  38000 #ton
        # displacement = 200000
        added_mass = 2660
        # added_mass = 200000 * 2625 / 37500
        brake_force = 14.1
        brake_per = brake_force / (displacement + added_mass)
        
        safety_margin = [0, 0.3, 0.6, 0.95]
    
    speed_list = []
    speed_list_2 = [[],[],[],[]] 
    speed_hat_list = [[], [], [], []]
    #---------------------------------------------------------------------------------------------------------------------------------    
    
    flag = 1
    
    if flag == 0:
        for i in range(len(berth_distance)):
            speed = speed_reducation(V_N, Lpp, berth_distance[i], sensitivity_coef, constant_integration)
            speed_list.append(speed)
       
        plt.plot(berth_distance / Lpp, speed_list)
        plt.xlabel("berth distance", fontname="MS Gothic")
        plt.ylabel("V", fontname="MS Gothic")
        plt.title("speed_reducation", fontname="MS Gothic")
        plt.savefig("speed_reducation.pdf")
    else:
        for j in range(len(safety_margin)):
            for i in range(len(berth_distance)):
                speed = speed_reducation_guideline(berth_distance[i], Lpp, V_N, displacement, added_mass, brake_force, safety_margin[j])
                knot_speed = speed * 3600 / 1852
                speed_list_2[j].append(knot_speed)
                speed_hat_list[j].append(speed / np.sqrt(g * Lpp))
        
            ax.plot(berth_distance/Lpp, speed_hat_list[j], label=f'safety margin:{safety_margin[j]}')
        
        plt.xlabel("berth distance / Lpp")
        plt.ylabel("V / √g*L")
        plt.title("speed_reducation_guideline")
        plt.legend()
        ax.fill_between(berth_distance / Lpp, speed_hat_list[1], speed_hat_list[2], facecolor='lime')
        plt.savefig(log_dir + "speed_reducation_guideline" + "brake" + f"{brake_force}" + "GT" + f'{displacement}' + ".pdf")
  
def nonlinear_fit(x,a,b):
        return a*x/(b+x)
            
            
            
            
            
def guideline_origin(data_list, log_dir, file_name, ship_flag="VLCC"):
    if ship_flag == "VLCC":
        Lpp = 324
    
        berth_distance_03_06 = np.arange(0, 10000, 1000)
        berth_distance_1 = np.arange(0, 10000, 1000)
        berth_distance_red = np.arange(0, 4000, 1000)
        knot_safety_03 = np.array([0, 2, 3, 3.9, 4.4, 5, 5.55, 6.18, 6.72, 7.46])
        knot_safety_06 = np.array([0, 2.9, 4.2, 5.2, 6.2, 7.36, 8.36, 9.68, 10.8, 10.8])
        knot_safety_1 = np.array([0, 3.6, 5.1, 6.6, 8, 9.9, 10.8, 12, 13, 14])
        knot_safety_red = np.array([0, 6.8, 9.8, 12])
        x_curveplot = np.linspace(0, 10000,100)
        
        V_N = 14 * 1852/3600
        
    elif ship_flag == "container":
        Lpp = 230
    
        berth_distance_03_06 = np.arange(0, 3500, 500)
        berth_distance_1 = np.arange(0, 2500, 500)
        berth_distance_red = np.arange(0, 1500, 500)
        knot_safety_03 = np.array([0, 2.2, 3.4, 4.1, 5, 5.8, 6.4])
        knot_safety_06 = np.array([0, 3.4, 4.9, 6.2, 7.9, 9.1, 10.8])
        knot_safety_1 = np.array([0, 4.1, 6, 8.1, 10.2])
        knot_safety_red = np.array([0, 6.7, 9.9])
        x_curveplot = np.linspace(0, 3000,100)
    
    elif ship_flag == "LNG":
        Lpp = 283
    
        berth_distance_03_06 = np.arange(0, 3500, 500)
        berth_distance_1 = np.arange(0, 2500, 500)
        berth_distance_red = np.arange(0, 1500, 500)
        knot_safety_03 = np.array([0, 2.2, 3.4, 4.1, 5, 5.8, 6.4])
        knot_safety_06 = np.array([0, 3.4, 4.9, 6.2, 7.9, 9.1, 10.8])
        knot_safety_1 = np.array([0, 4.1, 6, 8.1, 10.2])
        knot_safety_red = np.array([0, 6.7, 9.9])
        x_curveplot = np.linspace(0, 3000,100)
        
    elif ship_flag == "PCC":
        Lpp = 170
    
        berth_distance_03_06 = np.arange(0, 3500, 500)
        berth_distance_1 = np.arange(0, 2500, 500)
        berth_distance_red = np.arange(0, 1500, 500)
        knot_safety_03 = np.array([0, 2.2, 3.4, 4.1, 5, 5.8, 6.4])
        knot_safety_06 = np.array([0, 3.4, 4.9, 6.2, 7.9, 9.1, 10.8])
        knot_safety_1 = np.array([0, 4.1, 6, 8.1, 10.2])
        knot_safety_red = np.array([0, 6.7, 9.9])
        x_curveplot = np.linspace(0, 3000,100)
            
    hat_flag = "V_N"
    
    if hat_flag == "L*g":
        hat_speed_safety_03 = knot_safety_03 / np.sqrt(Lpp * 9.8)
        hat_speed_safety_06 = knot_safety_06 / np.sqrt(Lpp * 9.8)
        hat_speed_safety_1 = knot_safety_1 / np.sqrt(Lpp * 9.8)
        hat_speed_safety_red = knot_safety_red / np.sqrt(Lpp * 9.8)
    elif hat_flag == "V_N":
        hat_speed_safety_03 = knot_safety_03 / V_N
        hat_speed_safety_06 = knot_safety_06 / V_N
        hat_speed_safety_1 = knot_safety_1 / V_N
        hat_speed_safety_red = knot_safety_red / V_N
    
    #curve fittingの実施
    param_03, cov_03 = curve_fit(nonlinear_fit, berth_distance_03_06, hat_speed_safety_03)
    param_06, cov_06 = curve_fit(nonlinear_fit, berth_distance_03_06, hat_speed_safety_06)
    param_1, cov_1 = curve_fit(nonlinear_fit, berth_distance_1, hat_speed_safety_1)  
    param_red, cov_red = curve_fit(nonlinear_fit, berth_distance_red, hat_speed_safety_red)  
    
    # x_curveplot = np.linspace(0, 3000, 100)
    y_curveplot_03 = param_03[0]*x_curveplot/(param_03[1] + x_curveplot)
    y_curveplot_06 = param_06[0]*x_curveplot/(param_06[1] + x_curveplot)
    y_curveplot_1 = param_1[0]*x_curveplot/(param_1[1] + x_curveplot)
    bottom_line = np.zeros(len(x_curveplot))
    top_line = np.ones(len(x_curveplot)) * 10.8
    if hat_flag == "L*g":
        hat_top_line = top_line / np.sqrt(Lpp * 9.8)
    elif hat_flag == "V_N":
        hat_top_line = top_line / V_N
    
    # guidelineのプロット------------------------------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize = (7*1.2,5*1.2))
    # ax.plot(x_curveplot, param_03[0]*x_curveplot/(param_03[1] + x_curveplot),\
    # ls = '-', linewidth = 1 , color = 'black')
    # ax.plot(x_curveplot, param_06[0]*x_curveplot/(param_06[1] + x_curveplot),\
    # ls = '-', linewidth = 1 , color = 'black')
    # ax.plot(x_curveplot, param_1[0]*x_curveplot/(param_1[1] + x_curveplot),\
    # ls = '-', linewidth = 1 , color = 'black')
    # ax.plot(x_curveplot, param_red[0]*x_curveplot/(param_red[1] + x_curveplot),\
    # ls = '-', linewidth = 1 , color = 'black')
    
    ax.plot(x_curveplot / Lpp, param_03[0]*x_curveplot/(param_03[1] + x_curveplot),\
    ls = '-', linewidth = 1 , color = 'black')
    ax.plot(x_curveplot / Lpp, param_06[0]*x_curveplot/(param_06[1] + x_curveplot),\
    ls = '-', linewidth = 1 , color = 'black')
    ax.plot(x_curveplot / Lpp, param_1[0]*x_curveplot/(param_1[1] + x_curveplot),\
    ls = '-', linewidth = 1 , color = 'black')
    ax.plot(x_curveplot / Lpp, param_red[0]*x_curveplot/(param_red[1] + x_curveplot),\
    ls = '-', linewidth = 1 , color = 'black')
    # ax.plot(berth_distance / L_vlcc, hat_speed_safety_03, marker = 'o', markersize = 6, ls= '', color = 'black')
    # ax.plot(berth_distance / L_vlcc, hat_speed_safety_06, marker = 'o', markersize = 6, ls= '', color = 'black')
    # ax.plot(berth_distance/ L_vlcc, hat_speed_safety_1, marker = 'o', markersize = 6, ls= '', color = 'black')
    
    # ax.fill_between(x_curveplot, bottom_line , param_03[0]*x_curveplot/(param_03[1] + x_curveplot),\
    # alpha=0.6, facecolor='blue', label="available")
    # ax.fill_between(x_curveplot, param_03[0]*x_curveplot/(param_03[1] + x_curveplot), param_06[0]*x_curveplot/(param_06[1] + x_curveplot),\
    # alpha=0.5, facecolor='lime', label="recommendable")
    # ax.fill_between(x_curveplot, param_06[0]*x_curveplot/(param_06[1] + x_curveplot), param_1[0]*x_curveplot/(param_1[1] + x_curveplot),\
    # alpha=0.6, facecolor='blue', label="available")
    # ax.fill_between(x_curveplot, param_1[0]*x_curveplot/(param_1[1] + x_curveplot), param_red[0]*x_curveplot/(param_red[1] + x_curveplot),\
    # alpha=0.5, facecolor='red', label="amber color")
    # ax.fill_between(x_curveplot, param_red[0]*x_curveplot/(param_red[1] + x_curveplot), top_line,\
    # alpha=1, facecolor='red', label="red color")
    
    ax.fill_between(x_curveplot / Lpp, bottom_line , param_03[0]*x_curveplot/(param_03[1] + x_curveplot),\
    alpha=0.6, facecolor='blue')
    ax.fill_between(x_curveplot / Lpp, param_03[0]*x_curveplot/(param_03[1] + x_curveplot), param_06[0]*x_curveplot/(param_06[1] + x_curveplot),\
    alpha=0.5, facecolor='lime')
    ax.fill_between(x_curveplot / Lpp, param_06[0]*x_curveplot/(param_06[1] + x_curveplot), param_1[0]*x_curveplot/(param_1[1] + x_curveplot),\
    alpha=0.6, facecolor='blue')
    ax.fill_between(x_curveplot / Lpp, param_1[0]*x_curveplot/(param_1[1] + x_curveplot), param_red[0]*x_curveplot/(param_red[1] + x_curveplot),\
    alpha=0.5, facecolor='red')
    ax.fill_between(x_curveplot / Lpp, param_red[0]*x_curveplot/(param_red[1] + x_curveplot), top_line,\
    alpha=0.7, facecolor='red')
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # 計算結果のプロット------------------------------------------------------------------------------------------------------------------------------------------
    for i, data in enumerate(data_list):
        # print(data)
        # for index, row in data.iterrows():
        
        g = 9.8
        L = 3
        L_150 = 150
        B = 0.48925
        esso_V_N = 2.77 * 1852/3600 #(knot)
        x = data['x_position_mid [m]']
        y = data['y_position_mid [m]']
        u = data['u_velo [m/s]']
        psi = data['psi [rad]']
        # print(x, y ,u)
        x_fin = np.array([0.1-L, 0, -0.1+(-0.9)*B, 0, np.pi, 0]).T
        x_berth = x_fin[0]   
        y_berth = x_fin[1]
        berth_distance_array = np.sqrt((x - x_berth)**2 + (y - y_berth)**2)
        berth_distance_array_hat = berth_distance_array / L
        # berth_distance_array = berth_distance_array * 3600/1852
        knot_u = u * 3600/1852
        # knot_u_150 = knot_u * np.sqrt(L_150/L)
        # esso_speed_hat = knot_u / esso_V_N 
        if hat_flag == "L*g":
            esso_speed_hat = u / np.sqrt(g * L)
        if hat_flag == "V_N":    
            esso_speed_hat = u / esso_V_N
        # print(berth_distance_array)
        ax.plot(berth_distance_array_hat, esso_speed_hat, linewidth=2, linestyle=line_list[i], color="black", label=data_name[i])
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # file_count = 1
    # directory_path = "my_src/kaiseki/data"
    # for dirpath, dirnames, filenames in os.walk(directory_path):
    #     # '運動'を含む名前のフォルダを見つけた場合
    #     for dirname in dirnames:
    #         if '着' in dirname:
    #             # '運動'フォルダ内の'pos'を含む名前のフォルダを見つけます
    #             for sub_dirpath, sub_dirnames, sub_filenames in os.walk(os.path.join(dirpath, dirname)):
    #                 for sub_dirname in sub_dirnames:
    #                     if 'pos' in sub_dirname:
    #                         # 'pos'フォルダ内のすべての.xyzファイルを見つけます
    #                         for filename in os.listdir(os.path.join(sub_dirpath, sub_dirname)):   
    #                             if filename.endswith('.xyz'):
    #                                 xyz_file_path = os.path.join(sub_dirpath, sub_dirname, filename)
    #                             if filename.endswith('.his2'):
    #                                 his_file_path = os.path.join(sub_dirpath, sub_dirname, filename)
                                
    #                         ship_position, v_knots = test.process_file(xyz_file_path, his_file_path, file_count)
    #                         # データフレームの内容を表示します
    #                         ax.plot(ship_position, v_knots, label=f"{file_count}")
    #                         file_count += 1 
    #保存--------------------------------------------------------------------------------------------------------------------------------------------
    ax.grid()
    # ax.set_ylim(bottom=0, top=10.8)
    if hat_flag == "L*g":
        ax.set_ylim(bottom=0, top=10.8/np.sqrt(Lpp * 9.8))
    if hat_flag == "V_N":
        ax.set_ylim(bottom=0, top=10.8/V_N)
    # ax.set_xlim(left=0, right=3000)
    ax.set_xlim(left=0, right=3000/Lpp)
    ax.set_xlabel("$berth distance / Lpp$")
    if hat_flag == "V_N":
        ax.set_ylabel("$V / V_{\mathrm{N}}$")
    ax.text(8-0.5, 0.2, '$avaidable$')
    ax.text(7-0.5, 0.5, '$recommemdable$')
    ax.text(6-0.5, 0.9, '$amber color$')
    ax.text(2-0.5, 1, '$red color$')
    ax.legend(loc=1, fontsize=12)
    fig.savefig(log_dir + file_name + ship_flag + ".pdf")
    
def emg_stop_sim(data_list, stop_case_flag_list):
    multiple_x_stop_point_list =[]
    for j, data in enumerate(data_list):
        x_stop_point_list = []
        x_position_mid = data['x_position_mid [m]']
        u_velo = data['u_velo [m/s]']
        y_position_mid = data['y_position_mid [m]']
        vm_velo = data['vm_velo [m/s]']
        psi = data['psi [rad]']
        r_angvelo = data['r_angvelo [rad/s]']
        delta_rudder = data['delta_rudder [rad]']
        n_prop = data['n_prop [rps]']
        true_wind_speed = data['true_wind_speed [m/s]']
        true_wind_direction = data['true_wind_direction [rad]']
        for i in range(int(len(data['delta_rudder [rad]'])/100 - 0.5)):
            state = np.array([
            x_position_mid[i*100], u_velo[i*100], y_position_mid[i*100], vm_velo[i*100], psi[i*100], r_angvelo[i*100],
            delta_rudder[i*100], n_prop[i*100],
            true_wind_speed[i*100], true_wind_direction[i*100]
            ])
            
            x_stop_point_list.append(state)
    
        multiple_x_stop_point_list.append(np.array(x_stop_point_list))
        
    # multiple_x_stop_point_array = np.array(multiple_x_stop_point_list))
    
    x_hat_list0 = []
    for i in range(len(data_list)):
        x_hat_list0.append([])
    count = 0
    for j, x_stop_point_list_traj in enumerate(multiple_x_stop_point_list):
        x_hat = np.full((len(x_stop_point_list_traj), 6), np.nan)
        print(x_stop_point_list_traj[:, 4])
        # x_stop_point_list_traj[:, 4] *= 0.99
        print(x_stop_point_list_traj[:, 4])
        stop_case_flag = stop_case_flag_list[j]
        for i in range(len(x_stop_point_list_traj)):
            # print(x_stop_point_list[i])
            col_sim = Collision_Sim(x_stop_point_list_traj[i])
            col_sim()
            if stop_case_flag == "full":
                emg_x = col_sim.emg_stop_full(count)  
            elif stop_case_flag == "half":
                emg_x = col_sim.emg_stop_half(count)
            elif stop_case_flag == "rudder_stopped":
                emg_x = col_sim.emg_stop_rudder_stopped(count)
            else:
                emg_x = np.full((100, 6), np.nan)
            
            x_hat_list0[j].append(emg_x)
            count += 1

    return np.array(x_hat_list0)

    
def obstacle_distance(x_stop_point_list):
    path_berth, path_inukai = mat_obstacle()
    all_distance_list = [[], [], [], []]
    all_v_knot_list = [[], [], [], []]
    for ship_state_list in x_stop_point_list:
        distance_list = [[], [], [], []]
        v_knot_list = [[], [], [], []]
        for ship_state in ship_state_list:
            for angle_offset in [0, np.pi/2, -np.pi/2, np.pi]:  # 0: 前方向, pi/2: 左方向, -pi/2: 右方向, pi: 後方向
                psi = ship_state[4] + angle_offset
                b = -np.tan(psi) * ship_state[0] + ship_state[2]
                print(ship_state[1], ship_state[3])
                # if np.pi/2 <= psi <=np.pi*3/2:
                #     x_array = np.arange(ship_state[0], -40, 0.1)
                # else:
                #     x_array = np.arange(ship_state[0], 100, 0.1)
                x_array = np.arange(ship_state[0], -40, -0.1)
                print(x_array)
                for x in x_array:
                    y = np.tan(psi) * x + b
                    point = (x, y)
                    berth_mask = path_berth.contains_point(point)
                    inukai_mask = path_inukai.contains_point(point)
                    
                    combined_mask = berth_mask | inukai_mask  # 領域内の点を示すマスク
                    intercect_flag = bool(combined_mask)
                    if intercect_flag:
                        distance = np.sqrt((ship_state[0] - point[0])**2 + (ship_state[2] - point[1])**2)
                        velocity = np.sqrt(ship_state[1]**2 + ship_state[3]**2)
                        v_knot = velocity * 3600/1852
                        distance_list[i].append(distance.copy())
                        v_knot_list[i].append(v_knot.copy()) 
                        break
        
        all_distance_list.append(distance_list)
        all_v_knot_list.append(v_knot_list)
                        
        
    print(distance_list)
    fig, ax2 = plt.subplots()
    ax2.grid()
    for distances, v_knots in zip(all_distance_list, all_v_knot_list):
        ax2.plot(distances, v_knots)
    fig.savefig(log_dir + "front_separation_distance.pdf")
    

if __name__=="__main__":
    # ------------------------------------------------------------------------------------------------------------------------------------------
    fin_case_flag ="fin_1"
    stop_case_flag ="emg_stop"
    number="0"
    prefix = f"{fin_case_flag}_{stop_case_flag}_{number}"
    log_dir = "result/"
    csv_dir = log_dir + f"{prefix}.csv"
    data = pd.read_csv(csv_dir)
    # ---------------------------------------------- --------------------------------------------------------------------------------------------
    data1 = pd.read_csv(log_dir + "fin_1_no_stop_1.csv")
    data2 = pd.read_csv(log_dir + "fin_1_emg_stop_0.csv")
    data3 = pd.read_csv(log_dir + "fin_1_half_0.csv")
    data4 = pd.read_csv(log_dir + "fin_4_no_stop_1.csv")
    data5 = pd.read_csv(log_dir + "sample.csv")
    # data5 = pd.read_csv(log_dir + "fin_2_no_stop.csv")c
    # data5 = pd.read_csv(log_dir + "fin_2_full.csv")
    prefix_list = ["fin_1_no_stop_1", "fin_1_emg_stop_0", "fin_1_half_0", "fin_4_no_stop_1", "sample"]
    line_list = ["-", "--", ":", "-.", "-"]
    data_name = ["case1", "case2", "case3", "case4", "case5"]
    stop_case_flag_list = ["no_stop", "full", "half", "no_stop", "full"]
    # data4 = pd.read_csv(log_dir + "init3_stop3_0.csv")
    # data5 = pd.read_csv(log_dir + "init3_stop2_0.csv")
    data_list = [data1, data2, data3, data4, data5]
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    ship=shipsim.EssoOsaka_xu()  
    # Generation mode can be selected from "random" or "stationary".
    wind = shipsim.Wind(mode="stationary")
    world=shipsim.InukaiPond(wind=wind)
    sim = shipsim.ManeuveringSimulation(
        ship=ship,
        world=world,
        dt_act=1.0,
        dt_sim=0.1,
        solve_method="rk4", # "euler" or "rk4"
        log_dir=log_dir,
        check_collide=False,
    )
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # t [s],x_position_mid [m],u_velo [m/s],y_position_mid [m],vm_velo [m/s],psi [rad],r_angvelo [rad/s],delta_rudder [rad],n_prop [rps],true_wind_speed [m/s],true_wind_direction [rad],delta_rudder_cmd [rad],n_prop_cmd [rps],x_position_mid_hat [m],u_velo_hat [m/s],y_position_mid_hat [m],vm_velo_hat [m/s],psi_hat [rad],r_angvelo_hat [rad/s],delta_rudder_hat [rad],n_prop_hat [rps],true_wind_speed_hat [m/s],true_wind_direction_hat [rad],collision,apparent_wind_speed [m/s],apparent_wind_direction [rad],apparent_wind_speed_hat [m/s],apparent_wind_direction_hat [rad]
    
    x_hat_list = emg_stop_sim(data_list, stop_case_flag_list)
    # ---------------------------------------------------------------------------------------------------------------------
    # init_psi_list = []
    # init_speed_list = []
    # init_x_list = []
    # init_y_list = []
    # distance_list = []
    # for i in range(len(x_stop_point_list_traj)):
    #     init_x = x_stop_point_list_traj[i, 0]
    #     init_y = x_stop_point_list_traj[i, 2]
    #     init_u = x_stop_point_list_traj[i, 1]
    #     init_v_m = x_stop_point_list_traj[i, 3]
    #     init_psi = x_stop_point_list_traj[i, 4]
    #     init_speed = np.sqrt(init_u**2 + init_v_m**2)
    #     last_x = x_hat_array[i, -1, 0]
    #     last_y = x_hat_array[i, -1, 2]
    # # last_x_hat_array =  x_hat_array[]
    #     distance = np.sqrt((init_x - last_x)**2 + (init_y - last_y)**2)
    #     init_x_list.append(init_x)
    #     init_y_list.append(init_y)
    #     init_psi_list.append(init_psi)
    #     init_speed_list.append(init_speed)
    #     distance_list.append(distance)
        
    # knot_speed_array = np.array(init_speed_list) * 3600 /1852
    # hat_distance_array = np.array(distance_list) / 3
    
    # plt.grid()
    # plt.plot(hat_distance_array, knot_speed_array)
    # plt.savefig("result/emg_stop_distance.pdf")
    # # ----------------------------------------------------------------------------------------------------------------------
    # U_min_150 = 1#knot
    # U_max_150 = 6#kont
    
    # U_min_3 = U_min_150 * 1852/3600 * np.sqrt(3/150)
    # U_max_3 = U_max_150 * 1852/3600 * np.sqrt(3/150)
    
    # # a = L_x(U) +0.5 L
    # # b = L_y(U) +0.5 B
        
    # L = 3
    # B = 0.48925
    # W = 3.04 *  L
    
    # L_x_max_L = 0.75 * W - 0.5 * L
    # L_x_max_S = 0.5 * W - 0.5 * L
    # L_y_max = 0.25 * W - 0.5 * L
    
    # L_x_min = 0.25 * L
    # L_y_min = B
    
    # P_x_list = []
    # P_y_list = []    
    # p_array_list = np.zeros((len(init_speed_list), 13, 2))
    # for j, U in enumerate(init_speed_list): 
    #     P_x = np.zeros(14)
    #     P_y = np.zeros(14)
    #     psi = init_psi_list[j]
    #     R = np.array([[np.cos(psi), -np.sin(psi)],
    #                 [np.sin(psi), np.cos(psi)]])
    #     x = init_x_list[j]
    #     y = init_y_list[j]
    #     if U > U_max_3:
    #         L_x_L = L_x_max_L
    #     elif U_min_3 < U < U_max_3:
    #         L_x_L = L_x_min + (L_x_max_L - L_x_min) * (U - U_min_3) / (U_max_3 - U_min_3)
    #     elif U < U_min_3:
    #         L_x_L = L_x_min
            
    #     if U > U_max_3:
    #         L_x_S = L_x_max_S
    #     elif U_min_3 < U < U_max_3:
    #         L_x_S = L_x_min + (L_x_max_S - L_x_min) * (U - U_min_3) / (U_max_3 - U_min_3)
    #     elif U < U_min_3:
    #         L_x_S = L_x_min
        
    #     if U > U_max_3:
    #         L_y = L_y_max
    #     elif U_min_3 < U < U_max_3:
    #         L_y = L_y_min + (L_y_max - L_y_min) * (U  - U_min_3) / (U_max_3 - U_min_3)
    #     elif U < U_min_3:
    #         L_y = L_y_min
        
    #     for i in range(13):
    #         alpha = i / 12 * 2 * np.pi
    #         if x_stop_point_list_traj[j, 1] >= 0:
    #             if - np.pi / 2 < alpha < np.pi / 2:
    #                 L_x = L_x_L
    #             else:
    #                 L_x = L_x_S
    #         else:
    #             if - np.pi / 2 < alpha < np.pi / 2:
    #                 L_x = L_x_S
    #             else:
    #                 L_x = L_x_L 
                                     
    #         a = L_x + 0.5 * L
    #         b = L_y + 0.5 * B
    #         print("a", a)
            
    #         P_x = a * np.cos(alpha)
    #         P_y = b * np.sin(alpha)
    #         print(P_x, P_y)
    #         P = np.array([P_x, P_y])
    #         p_array = np.dot(R, P)
    #         p_array_list[j][i][0] = x + p_array[0]
    #         p_array_list[j][i][1] = y + p_array[1]
    #     P_x_list.append(P_x)
    #     P_y_list.append(P_y)
    #     p_array_list[j][12][0] = p_array_list[j][0][0]
    #     p_array_list[j][12][1] = p_array_list[j][0][1]
        
    # p_x_array = np.array(P_x_list)
    # p_y_array = np.array(P_y_list)
    
    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    plot.action_plot(data_list[0:3], line_list, log_dir, prefix="48")       
    plot.state_plot(data_list[0:3], line_list, log_dir, prefix="48") 
    plot.delta_plot(data_list[0], line_list, log_dir, prefix="3")
    
    
    
    # ---------------------------------------------------------------------------------------------------------------------
    for i, x_hat in enumerate(x_hat_list):
        x_hat_array = np.array(x_hat) 
        prefix = prefix_list[i]
        csv_dir = log_dir + f"{prefix}.csv"
        sim.viewer.get_x0y0_plot_2(prefix=prefix + "_size048", csv_dir=csv_dir, p_array=None,  x_emg_stop=x_hat_array, ext_type="pdf")
    # sim.viewer.get_timeseries_plot(TIME_NAME=["t [s]"], NAMES= , prefix="state")
    sim.viewer.get_x0y0_plot_2(prefix="inukai", csv_dir=csv_dir, p_array=None,  x_emg_stop=None, ext_type="pdf")
    
    # plot_guideline(data_list, log_dir)
    # obstacle_distance(multiple_x_stop_point_list)
    guideline_origin(data_list[0:3], log_dir, file_name='guideline_0228')
    print("Task Completed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")