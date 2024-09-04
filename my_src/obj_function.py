import numpy as np
from numba import jit

# @jit
def J_2(x_list, t_list, x_1, x_tol, collision_p_list, tf, w_2, y_berth):
    sum_term = 0
    x_last = x_list[-1]
    for i in range(6):
        max_term_i = max((x_last[i] - x_1[i])**2, x_tol[i]**2)
        sum_term += max_term_i
        # print(max_term_i, sum_term)
    
    tau = [i / t_list[-1] for i in t_list]
    integral = 0
    for i in range(1, len(x_list)):
        delta_tau = tau[i] - tau[i - 1]
        integral += 0.5 * (np.linalg.norm(x_list[i] - x_1**2) + np.linalg.norm(x_list[i - 1] - x_1**2)) * delta_tau
        
    p0_y = [item['p0'][1] for item in collision_p_list]
    p1_y = [item['p1'][1] for item in collision_p_list]
    p2_y = [item['p2'][1] for item in collision_p_list]
    p3_y = [item['p3'][1] for item in collision_p_list]
    collision_t_list = [item['t'] for item in collision_p_list]
    
    collision_p_list_2 = [p0_y, p1_y, p2_y, p3_y]
    
    C_sum = 0
   
    for i in range(4):
        for j in range(len(collision_t_list)):
            if collision_p_list_2[i][j-1] is not None and collision_p_list_2[i][j] is not None:
                delta_collision_t = collision_t_list[j] - collision_t_list[j-1] 
                C_sum += 0.5 * (abs(collision_p_list_2[i][j] - y_berth) + abs(collision_p_list_2[i][j-1] - y_berth)) * delta_collision_t
                
    if np.isnan(C_sum):
        C_sum = 0
        
    print(sum_term, tf, integral, w_2, C_sum)
    J_2 = sum_term * tf * integral + w_2 * C_sum
    
    return J_2



def mat_J_2(x_list, t_list, x_1, x_tol, collision_p_list, tf, w_2, y_berth):
    sum_term = 0
    x_last = x_list[-1]
    for i in range(6):
        max_term_i = max((x_last[i] - x_1[i])**2, x_tol[i]**2)
        sum_term += max_term_i
        # print(max_term_i, sum_term)
    
    tau = [i / t_list[-1] for i in t_list]
    integral = 0
    for i in range(1, len(x_list)):
        delta_tau = tau[i] - tau[i - 1]
        integral += 0.5 * (np.linalg.norm(x_list[i] - x_1**2) + np.linalg.norm(x_list[i - 1] - x_1**2)) * delta_tau
        
    # p0_y = [item['p0'][:, 1] for item in collision_p_list]
    # p1_y = [item['p1'][:, 1] for item in collision_p_list]
    # p2_y = [item['p2'][:, 1] for item in collision_p_list]
    # p3_y = [item['p3'][:, 1] for item in collision_p_list]
    p0_y = collision_p_list["p0"][1, :]
    p1_y = collision_p_list["p1"][1, :]
    p2_y = collision_p_list["p2"][1, :]
    p3_y = collision_p_list["p3"][1, :]
    collision_t_list = collision_p_list["t"][:]
    
    collision_p_list_2 = [p0_y, p1_y, p2_y, p3_y]
    # print("debug_collision_p_list_2", collision_p_list_2)
    # print("debug_collision_p_list_2[1][1]", collision_p_list_2[1][0])
    # print(len(collision_p_list_2[0]))
    
    
    C_sum = 0
   
    for i in range(4):
        for j in range(1, len(collision_p_list_2[0])):
            if collision_p_list_2[i][j-1] is not None and collision_p_list_2[i][j] is not None:
                # delta_collision_t = collision_t_list[j] - collision_t_list[j-1] 
                delta_collision_t = 1
                C_sum += 0.5 * (abs(collision_p_list_2[i][j-1] - y_berth) + abs(collision_p_list_2[i][j] - y_berth)) * delta_collision_t
                
    if np.isnan(C_sum):
        C_sum = 0
        
    print(sum_term, tf, integral, w_2, C_sum)
    J_2 = sum_term * tf * integral + w_2 * C_sum
    
    return J_2


@jit
def emg_stop_J_1(x_list, t_list, x_1, x_tol, p_array, tf, w_2, y_berth):
    sum_term = 0
    x_last = x_list[-1]
    for i in range(6):
        max_term_i = max((x_last[i] - x_1[i])**2, x_tol[i]**2)
        sum_term += max_term_i
        # print(max_term_i, sum_term)
    
    
    integral = 0
    tau = [i / t_list[-1] for i in t_list]
    for i in range(1, len(x_list)):
        delta_tau = tau[i] - tau[i - 1]
        integral += 0.5 * (np.linalg.norm(x_list[i] - x_1**2) + np.linalg.norm(x_list[i - 1] - x_1**2)) * delta_tau
        
    
    C_sum = 0
    # print(f"p_array:{p_array}")
    # print(f"p_array[0]:{p_array[0]}")
    # print(f"p_array[0][0]:{p_array[0][0]}")
    # print(f"p_array[0][0][0]:{p_array[0][0][0]}")
    # print(f"p_array[0][0][0][0]:{p_array[0][0][0][0]}")
    
    for i in range(len(p_array)):
        for j in range(len(p_array[0])):
            for k in range(1, len(p_array[0][1][0])):
                p_y = p_array[i][j][1][k-1]
                # print(f"p_array[i][j][1][k-1]:{p_y}")
                if not np.isnan(p_array[i][j][1][k-1]) and not np.isnan(p_array[i][j][1][k]):
                    delta_collision_t = 1
                    C_sum += 0.5 * (abs(p_array[i][j][1][k-1] - y_berth) + abs(p_array[i][j][1][k] - y_berth)) * delta_collision_t
                    # print(f"C_sum:{C_sum}")
   
    if np.isnan(C_sum):
        C_sum = 0
        
        
    print(sum_term, tf, integral, w_2, C_sum)
    J_2 = sum_term * tf * integral + w_2 * C_sum
    
    return J_2

def emg_stop_J_2(tf, x_1, x_fin, x_tol, p_array, p_array_col, w_dim, w_pen, w, y_berth):
    sum_error = 0
    error_array = np.zeros(6)
    for i in range(6):
        if abs(x_fin[i] - x_1[i])  <= x_tol[i]:
            error_array[i] = w_dim[i] * x_tol[i]**2
            sum_error += error_array[i]
        else:
            error_array[i] = w_pen * (x_fin[i] - x_1[i])**2
            sum_error += error_array[i]
            
    C_sum = 0
    # print(f"p_array:{p_array}")
    # print(f"p_array[0]:{p_array[0]}")
    # print(f"p_array[0][0]:{p_array[0][0]}")
    # print(f"p_array[0][0][0]:{p_array[0][0][0]}")
    # print(f"p_array[0][0][0][0]:{p_array[0][0][0][0]}")
    
    for i in range(len(p_array_col)):
        for j in range(len(p_array_col[0])):
            for k in range(1, len(p_array_col[0][1][0])):
                # if not np.isnan(p_array[i][j][1][k-1]) and not np.isnan(p_array[i][j][1][k]):
                if not np.isnan(p_array_col[i][j][1][k-1]) and not np.isnan(p_array[i][j][1][k]):
                    delta_collision_t = 1
                    C_sum += 0.5 * (abs(p_array_col[i][j][1][k-1] - y_berth) + abs(p_array[i][j][1][k] - y_berth)) * delta_collision_t
                    # print(f"C_sum:{C_sum}")
   
    if np.isnan(C_sum):
        C_sum = 0
    
    
        
    J_2 = w * C_sum + tf * sum_error 
    return J_2, C_sum, error_array

def J_3(x_1, x_fin, x_tol, w_dim, w_pen):
    sum_error = 0
    error_array = np.zeros(6)
    for i in range(6):
        if abs(x_fin[i] - x_1[i])  <= x_tol[i]:
            error_array[i] = w_dim[i] * x_tol[i]**2
            sum_error += error_array[i]
        else:
            error_array[i] = w_pen * (x_fin[i] - x_1[i])**2
            sum_error += error_array[i]
        
    # J_3 = tf * sum_error 
    return sum_error, error_array

### the below is original
# def J_3(tf, x_1, x_fin, x_tol, w_dim, w_pen):
#     sum_error = 0
#     error_array = np.zeros(6)
#     for i in range(6):
#         if abs(x_fin[i] - x_1[i])  <= x_tol[i]:
#             error_array[i] = w_dim[i] * x_tol[i]**2
#             sum_error += error_array[i]
#         else:
#             error_array[i] = w_pen * (x_fin[i] - x_1[i])**2
#             sum_error += error_array[i]
        
#     J_3 = tf * sum_error 
#     return J_3, error_array


    
    
       
    
    