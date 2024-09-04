import numpy as np
import matplotlib.pyplot as plt
from sympy.geometry import Point, Polygon
from numba import jit, prange

@jit
def collision_detection(p, poly_berth, poly_inukai):
    flag = False
    point = Point(p)
    check1 = poly_berth.encloses_point(point)
    check2 = poly_inukai.encloses_point(point)
    # check = obstacle.encloses_point(point)
    if check1 or check2:
        # print("coodinate:", p, "collide")
        flag = True
    
    return flag

@jit
def mat_collision_detection(points, path_berth, path_inukai):
    berth_mask = path_berth.contains_points(points.T)
    inukai_mask = path_inukai.contains_points(points.T)
    
    
    combined_mask = berth_mask | inukai_mask  # 領域内の点を示すマスク
    
    indices = np.where(combined_mask)[0]  # Trueのインデックスを取得
    # if any(indices):
    #     print("collision")
    return indices


# @jit(parallel=True)
# def collision_detection(p, tf, poly_berth, poly_inukai):
#     for j in prange(int(tf+1)):
#         for i in prange(4):
#             coodinate = (p[:, i])
#             flag = collision_detection(coodinate, poly_berth, poly_inukai)
#             # collision_p_dict = {
#             #     "t": sim.get_t(),
#             #     f"p{i}": coodinate if flag else None
#             # }
#             if flag:
#                 collision_p[i] = p[i]
#                 # print("berth_collision")
#             else:
#                 collision_p[i] = None
                
#         # 変更箇所
#         collision_p_dict = {"t":sim.get_t(), "p0":collision_p[0].copy(), "p1":collision_p[1].copy(), "p2":collision_p[2].copy(), "p3":collision_p[3].copy()}
#         collision_p_list.append(collision_p_dict)
        
#     return collision_p_list    