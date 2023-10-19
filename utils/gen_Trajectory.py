import numpy as np
import math
from math import pi as pi
import datetime
import matplotlib.pyplot as plt
import os

from UR5_kinematics import *


class sCurve_Params:
    def __init__(self, params):
        self.Pini = params['Pini']
        self.Pend = params['Pend']
        self.Amax = params['Amax']
        aavg = params['aavg']
        self.Vmax = params['Vmax']
        self.sampT = params['sampT']
        self.save_path = params['save_path']
        # Check if aavg is valid
        if np.any(aavg > 1) or np.any(aavg < 0.5):
            print("average of acceleration must within [0.5, 1]")
        else:
            self.Aavg = aavg * self.Amax
        self.eff = params['eff']

class Cmd:
    def __init__(self, cmd):
        self.P = cmd['P']
        self.V = cmd['V']
        self.A = cmd['A']
        self.J = cmd['J']

def gen_Scurve(params, coordinate = "Joint"):
    # 7 stage Scurve
    # ============== Input ==============
    # params
    #   Pini   : Initial Position
    #   Pend   : Goal Position
    #   Amax   : (Axis, 1) Max (Linear/Angular) Acceleration
    #   Aavg   : (Axis, 1) Average (Linear/Angular) Acceleration ((1/2 ~ 1)*Amax)
    #   Vmax   : (Axis, 1) Max (Linear/Angular) Velocity
    #   sampT  : Time interval
    #   save_path : If save the Cmd to .txt

    # ============= Output ==============
    # Cmd
    #   P : (tf/sampT, Axis) Position Command
    #   V : (tf/sampT, Axis) Velocity Command
    #   A : (tf/sampT, Axis) Acceration Command
    #   J : (tf/sampT, Axis) Jerk Command

    # Parameters
    Pini    = params.Pini
    Pend    = params.Pend
    Aavg    = params.Aavg
    Amax    = params.Amax
    Vmax    = params.Vmax
    sampT   = params.sampT
    if_Save = False

    Axis = len(Pini)
    S    = abs(Pend - Pini)

    Ta = np.zeros([Axis, 1])
    Tb = np.zeros([Axis, 1])
    Tc = np.zeros([Axis, 1])
    Ts = np.zeros([Axis, 1])

    # Check if the command is to save
    try:
        command_save_path = params.save_path
        if_Save = True
    except NameError:
        if_Save = False

    for i in range(Axis):
        if S[i] < (Vmax[i] * Vmax[i] / Aavg[i]) and S[i] != 0:
            Vmax[i] = math.sqrt(S[i] * Aavg[i])
        # Time Interval Length
        Ta[i] = Vmax[i] / Aavg[i]
        Tb[i] = 2 * Vmax[i] / Amax[i] - Ta[i]
        Tc[i] = (Ta[i] - Tb[i]) / 2
        Ts[i] = (S[i] - Vmax[i] * Ta[i]) / Vmax[i]

    t_idx = np.argmax(2 * Ta + Ts)

    # Time node
    t1 = Tc[t_idx]
    t2 = Tc[t_idx] + Tb[t_idx]
    t3 = Ta[t_idx]
    t4 = Ta[t_idx] + Ts[t_idx]
    t5 = Ta[t_idx] + Ts[t_idx] + Tc[t_idx]
    t6 = Ta[t_idx] + Ts[t_idx] + Tc[t_idx] + Tb[t_idx]
    t7 = Ta[t_idx] + Ts[t_idx] + Ta[t_idx]
    
    t_vals = np.arange(0, t7[0], sampT)
    num_points = len(t_vals)

    PCmd = np.zeros((num_points, Axis))
    VCmd = np.zeros((num_points, Axis))
    ACmd = np.zeros((num_points, Axis))
    JCmd = np.zeros((num_points, Axis))

    # check Axis to determine Jerk of each joint
    if Axis == 1:
        Jerk = Amax * np.sign(Pend - Pini) / Tc
    else:
        Jerk = (Pend - Pini) / (1/6 * t1 ** 3 + 1/6 * t2 ** 3 + 1/3 * t3 ** 3 + 1/6 * t4 ** 3 - 1/6 * t5 ** 3 - 1/6 * t6 ** 3
                    - 1/2 * t1 * t3 ** 2 - 1/2 * t2 * t3 ** 2 + t7 * (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 - 1/2 * t4 ** 2
                    + 1/2 * t5 ** 2 + 1/2 * t6 ** 2 + t1 * t3 + t2 * t3) + 1/2 * t7 ** 2 * (t4 - t5 - t6) + 1/6 * t7 ** 3)

    # Generate Scurve
    t = 0
    for i in range(num_points):
        if t < t1:
            JCmd[i, :] = Jerk
            ACmd[i, :] = Jerk * t
            VCmd[i, :] = Jerk * (1/2 * t ** 2)
            PCmd[i, :] = Pini + Jerk * (1/6 * t ** 3)
        elif t1 <= t < t2:
            JCmd[i, :] = 0
            ACmd[i, :] = Jerk * t1
            VCmd[i, :] = Jerk * (-1/2 * t1 ** 2 + t1 * t)
            PCmd[i, :] = Pini + Jerk * (1/6 * t1 ** 3 - 1/2 * t1 ** 2 * t + 1/2 * t1 * t ** 2)
        elif t2 <= t < t3:
            JCmd[i, :] = -Jerk
            ACmd[i, :] = Jerk * (-t + t1 + t2)
            VCmd[i, :] = Jerk * (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 + t1 * t + t2 * t - 1/2 * t ** 2)
            PCmd[i, :] = Pini + Jerk * (1/6 * t1 ** 3 + 1/6 * t2 ** 3 + (-1/2 * t1 ** 2 - 1/2 * t2 ** 2) * t
                                + 1/2 * (t1 + t2) * t ** 2 + (-1/6) * t ** 3)
        elif t3 <= t < t4:
            JCmd[i, :] = 0
            ACmd[i, :] = 0
            VCmd[i, :] = Jerk * (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 + t1 * t3 + t2 * t3)
            PCmd[i, :] = Pini + Jerk * (1/6 * t1 ** 3 + 1/6 * t2 ** 3 + 1/3 * t3 ** 3 - 1/2 * t1 * t3 ** 2 - 1/2 * t2 * t3 ** 2
                                + (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 + t1 * t3 + t2 * t3) * t)
        elif t4 <= t < t5:
            JCmd[i, :] = -Jerk
            ACmd[i, :] = Jerk * (-t + t4)
            VCmd[i, :] = Jerk * (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 - 1/2 * t4 ** 2 + t1 * t3 + t2 * t3 + t4 * t - 1/2 * t ** 2)
            PCmd[i, :] = Pini + Jerk * (1/6 * t1 ** 3 + 1/6 * t2 ** 3 + 1/3 * t3 ** 3 + 1/6 * t4 ** 3 - 1/2 * t1 * t3 ** 2 - 1/2 * t2 * t3 ** 2
                                + (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 - 1/2 * t4 ** 2 + t1 * t3 + t2 * t3) * t
                                + 1/2 * t4 * t ** 2 - 1/6 * t ** 3)
        elif t5 <= t < t6:
            JCmd[i, :] = 0
            ACmd[i, :] = Jerk * (t4 - t5)
            VCmd[i, :] = Jerk * (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 - 1/2 * t4 ** 2 + 1/2 * t5 ** 2 + t1 * t3 + t2 * t3 + t4 * t - t5 * t)
            PCmd[i, :] = Pini + Jerk * (1/6 * t1 ** 3 + 1/6 * t2 ** 3 + 1/3 * t3 ** 3 + 1/6 * t4 ** 3 - 1/6 * t5 ** 3 - 1/2 * t1 * t3 ** 2 - 1/2 * t2 * t3 ** 2
                                + (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 - 1/2 * t4 ** 2 + 1/2 * t5 ** 2 + t1 * t3 + t2 * t3) * t
                                + 1/2 * (t4 - t5) * t ** 2)
        elif t6 <= t <= t7:
            JCmd[i, :] = Jerk
            ACmd[i, :] = Jerk * (t + t4 - t5 - t6)
            VCmd[i, :] = Jerk * (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2 - 1/2 * t4 ** 2 + 1/2 * t5 ** 2 + 1/2 * t6 ** 2
                                + t1 * t3 + t2 * t3 + t4 * t - t5 * t - t6 * t + 1/2 * t ** 2)
            PCmd[i, :] = Pini + Jerk * (1/6 * t1 ** 3 + 1/6 * t2 ** 3 + 1/3 * t3 ** 3 + 1/6 * t4 ** 3 - 1/6 * t5 ** 3 - 1/6 * t6 ** 3
                                - 1/2 * t1 * t3 ** 2 - 1/2 * t2 * t3 ** 2 + (-1/2 * t1 ** 2 - 1/2 * t2 ** 2 - 1/2 * t3 ** 2
                                - 1/2 * t4 ** 2 + 1/2 * t5 ** 2 + 1/2 * t6 ** 2 + t1 * t3 + t2 * t3) * t
                                + 1/2 * (t4 - t5 - t6) * t ** 2 + 1/6 * t ** 3)
        t += sampT
    
    if if_Save:
        current_datetime = datetime.datetime.now()
        # filename = f"Scurve_{current_datetime.strftime('%Y%m%d_%H%M%S')}.txt"
        filename = command_save_path
        with open(filename, 'w') as file:
            for i in range(num_points-1):
                row = np.concatenate((PCmd[i, :], VCmd[i, :], ACmd[i, :], JCmd[i, :]))
                row_str = ' '.join(map(str, row))
                file.write(row_str + '\n')

    cmd = {
        'P': PCmd,
        'V': VCmd,
        'A': ACmd,
        'J': JCmd
    }
    
    if coordinate == "Cartesian":
        orien      = params.eff
        Trajectory = command_cartesian_to_joint(cmd, orien)
    else:
        Trajectory = Cmd(cmd)

    return Trajectory

def command_cartesian_to_joint(cmd, orien):
    c_PCmd = cmd['P'] 
    j_PCmd = np.zeros([c_PCmd.shape[0], 6, 8])
    for i in range(c_PCmd.shape[0]):
        desired_pos = rot_trans_to_matrix(euler_to_rot_matrix(orien), c_PCmd[i, :])
        j_PCmd[i, :, :] = IK(desired_pos)

    filename = "../data/input/command_joint.txt"
    with open(filename, 'w') as file:
        for i in range(c_PCmd.shape[0]):
            row = j_PCmd[i, :, 0]
            for j in range(7):
                row = np.concatenate((row, j_PCmd[i, :, j+1]))
            row_str = ' '.join(map(str, row))
            file.write(row_str + '\n')


def plotCmd(Cmd, params):
    # Plot the commands
    time_vals = np.arange(0, Cmd.P.shape[0]) * params.sampT
    Axis = Cmd.P.shape[1]

    # Position commands
    plt.figure(figsize=(10, 6))
    for axis in range(Axis):
        plt.plot(time_vals, Cmd.P[:, axis], label=f'Axis {axis + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.title('Position Commands')
    plt.show()

    # Velocity commands
    plt.figure(figsize=(10, 6))
    for axis in range(Axis):
        plt.plot(time_vals, Cmd.V[:, axis], label=f'Axis {axis + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()
    plt.title('Velocity Commands')
    plt.show()

    # Acceleration commands
    plt.figure(figsize=(10, 6))
    for axis in range(Axis):
        plt.plot(time_vals, Cmd.A[:, axis], label=f'Axis {axis + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.title('Acceleration Commands')
    plt.show()

    # Jerk commands
    plt.figure(figsize=(10, 6))
    for axis in range(Axis):
        plt.plot(time_vals, Cmd.J[:, axis], label=f'Axis {axis + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Jerk')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    params = {
        'Pini'      : np.array([ 0.6, -0.1, 0.245]),
        'Pend'      : np.array([ 0.6,  0.1, 0.245]),
        'Amax'      : np.array([ 1.0,  1.0, 1.0]),
        'aavg'      : np.array([ 0.6,  0.7, 0.7]),
        'Vmax'      : np.array([ 0.5,  0.5,  0.5]),
        'sampT'     : 0.001,
        'save_path' : '../data/input/command.txt.',
        'eff'       : np.array([0, pi, pi])
    }

    c_Param = sCurve_Params(params)
    c_Cmd   = gen_Scurve(c_Param, "Cartesian")

    # plotCmd(c_Cmd, c_Param)


