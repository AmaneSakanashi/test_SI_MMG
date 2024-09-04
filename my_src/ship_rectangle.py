import numpy as np
from numba import jit

@jit
def calculate_ship_rectangle(x, y, psi, L, B):
    half_L = L / 2
    half_B = B / 2
    
    rectangle_p_0 = np.array([
        [x + half_L, y - half_B],
        [x + half_L, y + half_B],
        [x - half_L, y + half_B],
        [x - half_L, y - half_B]
    ])
    
    rectangle_p_0 = np.array([x + half_L, y - half_B])
    
    
    # print(rectangle_p_0)
    # print(rectangle_p_0 - np.array([x, y]))
    rotation_matrix = np.array([[np.cos(psi), -np.sin(psi)],
                                [np.sin(psi), np.cos(psi)]])
    
    rotated_p0 = np.dot(np.array([half_L, -half_B]), rotation_matrix) + np.array([x, y])
    rotated_p1 = np.dot(np.array([half_L, half_B]), rotation_matrix) + np.array([x, y])
    rotated_p2 = np.dot(np.array([-half_L, half_B]), rotation_matrix) + np.array([x, y])
    rotated_p3 = np.dot(np.array([-half_L, -half_B]), rotation_matrix) + np.array([x, y])
    
    # print(rotated_p0)
    
    return rotated_p0, rotated_p1, rotated_p2, rotated_p3
    
    # rotated_points = (rectangle_points - np.array([x, y])).dot(rotation_matrix.T) + np.array([x, y])
    # rotated_points = np.dot(rectangle_points - np.array([x, y]), rotation_matrix.T) + np.array([x, y])
    
    # return tuple(rotated_points)

    # p_0 = np.array([x + half_L, y - half_B])
    # p_1 = np.array([x + half_L, y + half_B])
    # p_2 = np.array([x - half_L, y + half_B])
    # p_3 = np.array([x - half_L, y - half_B])

    # rotation_matrix = np.array([[np.cos(psi), -np.sin(psi)],
    #                             [np.sin(psi), np.cos(psi)]])

    # p_0 = np.dot(rotation_matrix, p_0 - np.array([x, y])) + np.array([x, y])
    # p_1 = np.dot(rotation_matrix, p_1 - np.array([x, y])) + np.array([x, y])
    # p_2 = np.dot(rotation_matrix, p_2 - np.array([x, y])) + np.array([x, y])
    # p_3 = np.dot(rotation_matrix, p_3 - np.array([x, y])) + np.array([x, y])

    # return p_0, p_1, p_2, p_3
