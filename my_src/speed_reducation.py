import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit
def speed_reducation(V_N, Lpp, berth_distance, sensitivity_coef, constant_integration):    
    # V 残存速力
    # V_N 定格速力
    # D 着岸点までの距離
    # Lpp　船の長さ
    # C　D/lpp = 1　のときのV/V_Nの値
    V_N = V_N
    Lpp = Lpp
    D = berth_distance 
    s_lambda = sensitivity_coef
    C = constant_integration
    
    speed = s_lambda * np.log(D/Lpp) + C
    
    return speed


@jit
def speed_reducation_guideline(berth_distance, Lpp, V_N, displacement, added_mass, brake_force, safety_margin):
    D = berth_distance
    Lpp = Lpp
    m = displacement
    m_x = added_mass
    F = brake_force
    R = safety_margin
    speed = np.sqrt(2 * D * (1 - R) * F * 9.8 / (m + m_x))
    # speed = F_r * np.sqrt(2 * (1 - R) * F / (m + m_x) * D / Lpp)
    
    return speed
    
    
if __name__ == '__main__':
    V_N = 16 
    Lpp = 200
    berth_distance = np.linspace(20*Lpp, 0) 
    
    sensitivity_coef = 0.109
    constant_integration = 0.15
    
    displacement =  38000 #ton
    # displacement = 200000
    added_mass = 2660
    # added_mass = 200000 * 2625 / 37500
    brake_force = 14.1
    brake_per = brake_force / (displacement + added_mass)
    print(brake_per)
    safety_margin = [0, 0.3, 0.6, 0.95]
    
    flag = 1
    
    speed_list = []
    speed_list_2 = [[],[],[],[]] 
    speed_hat_list = [[], [], [], []]
    
    fig, ax = plt.subplots()
    ax.grid()
    
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
                knot_speed = speed * 3600 / 1854
                speed_list_2[j].append(knot_speed)
                speed_hat_list[j].append(knot_speed / V_N)
        
            ax.plot(berth_distance/Lpp, speed_hat_list[j], label=safety_margin[j])
        plt.xlabel("berth distance / Lpp", fontname="MS Gothic")
        plt.ylabel("V / V_N", fontname="MS Gothic")
        plt.title("speed_reducation_guideline", fontname="MS Gothic")
        plt.legend()
        ax.fill_between(berth_distance / Lpp, speed_hat_list[1], speed_hat_list[2], facecolor='lime')
        plt.savefig("speed_reducation_guideline" + "brake" + f"{brake_force}" + "GT" + f'{displacement}' + ".pdf")
    
    
        
    print(speed_list)
    print("Task Completed")

