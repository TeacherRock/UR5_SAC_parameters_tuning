import random
import math
import numpy as np
import time
import random

from utils.vrep_func import *
from utils.UR5_kinematics import *

pi = math.pi

class Environment():
    def __init__(self, state_dim, action_dim):
        self.action_dim   = action_dim
        self.state_dim    = state_dim
        # self._action      = np.array([500, 500, 500, 500, 500, 500, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # self._action      = np.array([700, 700, 700, 700, 700, 700, 1, 1, 1, 1, 1, 1])
        self._action      = np.zeros([1, 12])
        self.upper_bound  = np.array([2000, 2000, 2000, 2000, 2000, 2000, 2, 2, 2, 2, 2, 2])
        self.lower_bound  = np.zeros([1, 12])
        self.clientID     = Vrep_connect()
        self.random_Goal  = True
        self.goal         = np.zeros([1, 6])

    def reset(self):
        # Reset the environment
        # Vrep_stop(self.clientID)
        Vrep_disconnect(self.clientID)
        self.clientID = Vrep_connect()
        self.random_Goal = True
        state = self.get_state()
        return state


    def step(self, action, ):
        # _action = self.NormalizeAction(action)
        # do action and return next_state, _reward, done
        self.NormalizeAction(action)
        
        if self.randomGoal:
            _goal = self.randomGoal()
        else:
            _goal = self.goal
        # _goal = [1.47164123,  0.33726598,  1.66184294, -0.42831264, -1.57079628,  1.47164123]
        
        # print("Action : ", _action)
        _reward, done = self.env_sim(_goal, self._action)

        # Next State
        next_state = self.get_state()

        # Reward calculate
        _reward = np.array(_reward)
        reward = self.cal_reward(_reward)

        done = True

        return next_state, reward, done

    def get_state(self):
        Vrep_start(self.clientID)
        _, _state, _, _ = Vrep_callLuafunction(self.clientID, 'get_State_Py')
        Vrep_pause(self.clientID)
        return _state
    
    def env_sim(self, _goal, _action):
        Vrep_start(self.clientID)
        # (np.array) to (list [[]]) to (list [])
        _action_ = _action.tolist()
        _action = []
        _action.extend(_action_[0])

        _, _, _, _ = Vrep_callLuafunction(self.clientID, 'set_Env_Py', self.save, _goal + _action, [], bytearray())
        time.sleep(2)

        Vrep_start(self.clientID)
        _ret, _reward, _, _ = Vrep_callLuafunction(self.clientID, 'return_Error_Py')
        _genScurve = _ret[0]
        _done      = _ret[1]
        Vrep_pause(self.clientID)

        if int(_done) > 10:
            done = 1
        elif _genScurve == 0:
            done = 1
            _reward = [0.03, 1.1]
        else:
            done = 0
        # Vrep_pause(self.clientID)
        Vrep_stop(self.clientID)
        return _reward, done

    def cal_reward(self, _reward):
        # for PD-like params
        avg_track_error = _reward[0]
        eff_error       = _reward[1]
        reward = 0

        print(f"Tracking Error : {avg_track_error} (rad), ", 
              f"Eff Error: {eff_error} (mm)")

        if eff_error > 20: # (mm)
            reward = reward - 10
        elif eff_error > 10:
            reward = reward - 5
        elif eff_error > 5:
            reward = reward + 5
        else:
            reward = reward + 10
        
        if avg_track_error > 0.03: # (rad)
            reward = reward - 10
        elif avg_track_error > 0.02:
            reward = reward - 5
        elif avg_track_error > 0.01:
            reward = reward + 2.5
        elif avg_track_error > 0.005: 
            reward = reward + 5
        else:
            reward = reward + 10

        return reward

    def randomGoal(self):
        # # Define the lower and upper bounds for each dimension
        # lower_bounds = [-pi, -pi, -pi, -pi, -pi, -pi]
        # upper_bounds = [ pi,  pi,  pi,  pi,  pi,  pi]

        # # Generate a random 6-dimensional float list
        # Pend = [random.uniform(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]

        # def check(Pend):
        #     XYZ = FK(np.matrix(Pend).T, 0)
        #     if XYZ[2] < 0:
        #         Pend = self.randomGoal()
        #     return Pend
        
        # Pend = check(Pend) 

        randomInt = random.randint(1, 11)

        if randomInt == 1:
            Pend = [1.47164123, -1.0, 1.76184294, -1.1, -1.57079628, -1.8164123]
        elif randomInt == 2:
            Pend = [1.47164123, 1.0, -1.56184294, 1, -1.57079628, 1.464123]
        elif randomInt == 3:
            Pend = [1.47164123, -1.0, 1.66184294, -1.034, -1.57079628, -1.47164123]
        elif randomInt == 4:
            Pend = [1.47164123, 1.0, -1.46184294, 1.3250, -1.57079628, 1.6164123]
        elif randomInt == 5:
            Pend = [-1.47164123, -1.0, 1.66184294, -1.350, -1.57079628, -1.6164123]
        elif randomInt == 6:
            Pend = [-1.47164123, 1.0, -1.86184294, -1.20, -1.57079628, 1.47164123]
        elif randomInt == 7:
            Pend = [-1.47164123, -1.0, 1.36184294, -1.40, -1.57079628, -1.5164123]
        elif randomInt == 8:
            Pend = [-1.47164123, 1.0, -1.66184294, 1.50, -1.57079628, -1.47164123]
        elif randomInt == 9:
            Pend = [-1.47164123, -1.0, 1.66184294, -1.90, -1.57079628, 1.7164123]
        else:
            Pend = [-1.47164123, 1.0, -1.66184294, 1.60, -1.57079628, -1.47164123]

        return Pend
    
    def NormalizeAction(self, action):
        # self._action  = self._action + action * self._action * 0.1
        # print("Type of action : ", type(action))

        # map the action from [-1, 1] to [0, 2500] and [0, 2.5]
        self._action[0, :6] = 2500 * (action[:6] + 1)/2
        self._action[0, 6:] = 2.5  * (action[6:] + 1)/2
        self._action = np.clip(self._action, self.lower_bound, self.upper_bound)
        # print("Kp : ", self._action[0, :6])
        # print("Kv : ", self._action[0, 6:])
        # print(self._action)
        # return self._action 

    def set_Goal(self, goal):
        self.random_Goal = False
        self.goal = goal
    
if __name__ == '__main__':
    env = Environment(2, 2)
    goal = env.randomGoal()
    print(goal)