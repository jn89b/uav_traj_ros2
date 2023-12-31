#!/usr/bin/env python3
import casadi as ca
import rclpy 
import numpy as np

#from drone_interfaces.msg import Telem, CtlTraj
from drone_interfaces.msg import Telem, CtlTraj, Waypoints
from uav_traj_ros2.CasadiModels.AirplaneModel import AirplaneSimpleModel
from uav_traj_ros2.MPC import MPC
from rclpy.node import Node
from uav_traj_ros2 import quaternion_tools, Config 

import pickle as pkl

import time

import mavros
from mavros.base import SENSOR_QOS

class Effector():
    def __init__(self, effector_range:float):
        self.effector_range = effector_range
        self.effector_power = 10

    def computePowerDensity(self,target_distance,
                            effector_factor):
        return self.effector_power * effector_factor/ (target_distance**2 * 4*ca.pi)

class AirplaneSimpleModelMPC(MPC):
    def __init__(self, mpc_params:dict, 
                 airplane_constraint_params:dict):
        super().__init__(mpc_params)
        self.airplane_params = airplane_constraint_params
        self.S = 0.5
        
        if 'effector' in self.airplane_params.keys():
            self.effector = self.airplane_params['effector']
            self.effector_range = self.effector.effector_range
            self.T = 0.1
        else:
            self.effector = None
            
    def computeCost(self):
        #tired of writing self
        #dynamic constraints 

        P = self.P
        Q = self.Q
        R = self.R
        n_states = self.n_states
        
        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]
            
            #penalize states and controls for now, can add other stuff too
            self.cost_fn = self.cost_fn \
                + (states - P[n_states:]).T @ Q @ (states - P[n_states:]) \
                + controls.T @ R @ controls                 

            # self.cost_fn =             
            ##Runge Kutta
            k1 = self.f(states, controls)
            k2 = self.f(states + self.dt_val/2*k1, controls)
            k3 = self.f(states + self.dt_val/2*k2, controls)
            k4 = self.f(states + self.dt_val * k3, controls)
            state_next_RK4 = states + (self.dt_val / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, state_next - state_next_RK4) #dynamic constraints
      
            if self.effector != None:
                x_pos = self.X[0,k]
                y_pos = self.X[1,k]
                z_pos = self.X[2,k]
                phi = self.X[3,k]
                theta = self.X[4,k]
                psi = self.X[5,k]

                dx = Config.GOAL_X - x_pos
                dy = Config.GOAL_Y - y_pos
                dz = Config.GOAL_Z - z_pos

                dtarget = ca.sqrt(dx**2 + dy**2 + dz**2)
                
                error_dist_factor = 1 - (dtarget / self.effector_range)
                
                effector_dmg = self.effector.computePowerDensity(
                    dtarget, error_dist_factor)
                
                #minus because we want to maximize damage
                self.cost_fn = self.cost_fn + (self.T* effector_dmg)

        if Config.RADAR_AVOID == True:
            for k in range(self.N):
                current_position = self.X[:3,k]
                current_heading = self.X[5,k]
                current_roll = self.X[4,k]

                detection_vals, prob_detection = self.radar.get_detection_probability(
                    current_position, current_heading, current_roll)

                self.cost_fn = self.cost_fn + (self.radar_weight * prob_detection)
                self.radar_history.append(detection_vals)

                #threshold
                detection_threshold = 0.5
                radar_constraint = prob_detection + detection_threshold

                self.g = ca.vertcat(self.g, radar_constraint)
                
        if Config.OBSTACLE_AVOID:
            for k in range(self.N):
                #penalize obtacle distance
                x_pos = self.X[0,k]
                y_pos = self.X[1,k]                
                obs_distance = ca.sqrt((x_pos - Config.OBSTACLE_X)**2 + \
                                        (y_pos - Config.OBSTACLE_Y)**2) 

    
                obs_constraint = -obs_distance + ((Config.ROBOT_DIAMETER/2) + \
                    (Config.OBSTACLE_DIAMETER/2))

                self.cost_fn = self.cost_fn - (self.S* obs_distance)

                self.g = ca.vertcat(self.g, obs_constraint) 

        if Config.MULTIPLE_OBSTACLE_AVOID:
            for obstacle in Config.OBSTACLES:
                obs_x = obstacle[0]
                obs_y = obstacle[1]
                obs_diameter = obstacle[2]

                for k in range(self.N):
                    #penalize obtacle distance
                    x_pos = self.X[0,k]
                    y_pos = self.X[1,k]                
                    obs_distance = ca.sqrt((x_pos - obs_x)**2 + \
                                            (y_pos - obs_y)**2)
                    

                    obs_constraint = -obs_distance + ((Config.ROBOT_DIAMETER) + \
                        (obs_diameter/2)) 

                    self.g = ca.vertcat(self.g, obs_constraint)
                
                    if obstacle == [Config.GOAL_X, Config.GOAL_Y]:
                        continue
                    
                    self.cost_fn = self.cost_fn + (self.S* obs_distance)


    def warmUpSolution(self, start:list, goal:list, controls:list) -> tuple:
        self.initDecisionVariables()
        self.defineBoundaryConstraints()
        self.addAdditionalConstraints()
        self.reinitStartGoal(start, goal)
        self.computeCost()
        self.initSolver()

        projected_controls,projected_states = self.solveMPCRealTimeStatic(
            start,goal, controls)
        
        return projected_controls,projected_states

    def addAdditionalConstraints(self) -> None:
        """add additional constraints to the MPC problem"""
        # add control constraints
        self.lbx['U'][0, :] = self.airplane_params['u_phi_min']
        self.ubx['U'][0, :] = self.airplane_params['u_phi_max']

        self.lbx['U'][1, :] = self.airplane_params['u_theta_min']
        self.ubx['U'][1, :] = self.airplane_params['u_theta_max']

        self.lbx['U'][2, :] = self.airplane_params['u_psi_min']
        self.ubx['U'][2, :] = self.airplane_params['u_psi_max']

        self.lbx['U'][3, :] = self.airplane_params['v_cmd_min']
        self.ubx['U'][3, :] = self.airplane_params['v_cmd_max']

        self.lbx['X'][2, :] = self.airplane_params['z_min']
        # self.ubx['X'][2,:] = self.airplane_params['z_max']

        self.lbx['X'][3, :] = self.airplane_params['phi_min']
        self.ubx['X'][3, :] = self.airplane_params['phi_max']

        self.lbx['X'][4, :] = self.airplane_params['theta_min']
        self.ubx['X'][4, :] = self.airplane_params['theta_max']

        self.lbx['X'][6, :] = self.airplane_params['airspeed_min']
        self.ubx['X'][6, :] = self.airplane_params['airspeed_max']

    def returnTrajDictionary(self, 
        projected_controls:list,
        projected_states:list) -> dict:
        traj_dictionary = {}
        traj_dictionary['x'] = projected_states[0,:]
        traj_dictionary['y'] = projected_states[1,:]
        traj_dictionary['z'] = projected_states[2,:]
        traj_dictionary['phi']= projected_states[3,:]
        traj_dictionary['theta'] = projected_states[4,:]
        traj_dictionary['psi'] = projected_states[5,:]
        traj_dictionary['v'] = projected_states[6,:]

        traj_dictionary['u_phi'] = projected_controls[0,:]
        traj_dictionary['u_theta'] = projected_controls[1,:]
        traj_dictionary['u_psi'] = projected_controls[2,:]
        traj_dictionary['v_cmd'] = projected_controls[3,:]

        return traj_dictionary
    
    def get_state_control_ref(self, 
        traj_dictionary:dict, 
        state_idx:int, 
        ctrl_idx:int) -> tuple:
        """get_state_control_ref"""
        x_ref = traj_dictionary['x'][state_idx]
        y_ref = traj_dictionary['y'][state_idx]
        z_ref = traj_dictionary['z'][state_idx]
        phi_ref = traj_dictionary['phi'][state_idx]
        theta_ref = traj_dictionary['theta'][state_idx]
        psi_ref = traj_dictionary['psi'][state_idx]
        vel_ref = traj_dictionary['v'][state_idx]

        u_phi_ref = traj_dictionary['u_phi'][ctrl_idx]
        u_theta_ref = traj_dictionary['u_theta'][ctrl_idx]
        u_psi_ref = traj_dictionary['u_psi'][ctrl_idx]
        v_cmd_ref = traj_dictionary['v_cmd'][ctrl_idx]

        return [x_ref, y_ref, z_ref, phi_ref, theta_ref, psi_ref, vel_ref], \
            [u_phi_ref, u_theta_ref, u_psi_ref, v_cmd_ref]

    def set_state_control_idx(self, mpc_params:dict, 
        solution_time:float, idx_buffer:int=0, use_buffer=False) -> int:
        """
        set index based on solution time
        Given the solution time round it to the nearest dt

        Based on that rounded value get the index of the control

        """
        #this rounds this to the nearest 0.1
        time_rounded = round(solution_time, 1)
        
        if time_rounded <= mpc_params['dt_val']:
            time_rounded = mpc_params['dt_val']
        
        control_idx = time_rounded/mpc_params['dt_val']

        if use_buffer:
            control_idx = control_idx + idx_buffer

        return int(round(control_idx))


class StateInfo():
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.z = None
        self.phi = None
        self.theta = None
        self.psi = None
        self.v = 15

class MPCTrajFWPublisher(Node):
    def __init__(self):
        super().__init__('mpc_traj_fw_publisher')

        self.get_logger().info('Starting MPC Traj FW Publisher')

        #turn this to a parameter later
        self.mpc_traj_freq = 100
        
        # self.state_info = [0,0,0,0,0,0,0,0] #x, y, z, psi, vx, vy, vz, psi_dot
        self.state_info = [Config.START_X, # x
                           Config.START_Y, # y
                           0, # z
                           0, # phi
                           0, # theta
                           np.deg2rad(180), # psi
                           18  # airspeed
                           ]
        
        self.control_info = [0, # u_phi
                             0, # u_theta
                             0, # u_psi
                             0  # v_cmd
                            ]
        
        self.traj_pub = self.create_publisher(CtlTraj, 
                        'trajectory', 
                        self.mpc_traj_freq)

        self.state_sub = self.create_subscription(mavros.local_position.Odometry,
                                                  'mavros/local_position/odom', 
                                                  self.positionCallback, 
                                                  qos_profile=SENSOR_QOS)

        self.wp_sub = self.create_subscription(Waypoints, "/global_waypoints", 
                                               self.waypointCallback, 10)
    
        self.initHistory()
        
        self.waypoint_list = []
        self.theta_dg_list = []
        self.psi_dg_list = []
        self.phi_dg_list = []

        self.found_new_path = False
        self.wp_buffer = 0
        self.wp_index = 0

    def waypointCallback(self,msg:Waypoints)->None:
        """
        Subscribes to waypoints in the message

        If waypoint lists are empty then append the waypoints

        if we have waypoints compare the current 
        waypoints and see if they are the same disregarding,
        the first waypoint or the second waypoint 
        """
        if len(self.waypoint_list) == 0:            
            for wp in msg.points:
                self.waypoint_list.append(wp)
                self.theta_dg_list = list(msg.pitch)
                self.psi_dg_list = list(msg.heading)
                self.phi_dg_list = list(msg.roll)

        # new path
        if self.waypoint_list != msg.points:
            self.found_new_path = True
            self.waypoint_list = []
            for wp in msg.points:
                self.waypoint_list.append(wp)
            self.theta_dg_list = list(msg.pitch)
            self.psi_dg_list = list(msg.heading)
            self.phi_dg_list = list(msg.roll)
        # not a new path
        else:
            self.found_new_path = False

        
    def get_next_goal_point(self) -> StateInfo:
        """
        get_next_goal_point returns the next goal point 
        based on the current waypoint index 

        If there is a new path, then we reset the waypoint index
        Otherwise we increment the waypoint index
        
        """
  
        if len(self.waypoint_list) == 0:
            print("We have no waypoints")
            return None
        
        state_info = StateInfo()
        
        if self.found_new_path == True:

            #need to do a checking mechanism here if I'm between
            if len(self.waypoint_list) == 1:
                state_info.x = self.waypoint_list[0].x
                state_info.y = self.waypoint_list[0].y
                state_info.z = self.waypoint_list[0].z
                state_info.phi = np.deg2rad(self.phi_dg_list[0])
                state_info.psi = np.deg2rad(self.psi_dg_list[0])
                state_info.theta = np.deg2rad(self.theta_dg_list[0])
                self.wp_index = self.wp_buffer
                return state_info
            # else:
            #     state_info.x = self.waypoint_list[self.wp_index].x
            #     state_info.y = self.waypoint_list[self.wp_index].y
            #     state_info.z = self.waypoint_list[self.wp_index].z
            #     state_info.phi = 0
            #     state_info.psi = np.deg2rad(self.psi_dg_list[self.wp_index+1])
            #     state_info.theta = np.deg2rad(self.theta_dg_list[self.wp_index+1])
            #     self.wp_index = self.wp_buffer
            #     return state_info
            
        state_info.x = self.waypoint_list[self.wp_index].x
        state_info.y = self.waypoint_list[self.wp_index].y
        state_info.z = self.waypoint_list[self.wp_index].z
        state_info.phi = np.deg2rad(self.phi_dg_list[self.wp_index+1])
        state_info.psi = np.deg2rad(self.psi_dg_list[self.wp_index+1])
        state_info.theta = np.deg2rad(self.theta_dg_list[self.wp_index+1])

        #this is a stupid way to trouble shoot but screw it
        # state_info.x = 50
        # state_info.y = 50
        # state_info.z = 50
        # state_info.phi = 0
        # state_info.psi = np.deg2rad(180)
        # state_info.theta = np.deg2rad(0)
        
        # next_wp = self.waypoint_list[self.wp_index+1]
        return state_info

    def initHistory(self) -> None:
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.phi_history = []
        self.theta_history = []
        self.psi_history = []
        self.v_history = []

        self.x_trajectory = []
        self.y_trajectory = []
        self.z_trajectory = []
        self.phi_trajectory = []
        self.theta_trajectory = []
        self.psi_trajectory = []
        self.idx_history = []
        self.v_trajectory = []
        self.obstacles = []
        self.radar_probability = []


        self.pickle_history = {
            'x_history': None,
            'y_history': None,
            'z_history': None,

            'phi_history': None,
            'theta_history': None,
            'psi_history': None,
            'v_history': None,

            'x_trajectory': None,
            'y_trajectory': None,
            'z_trajectory': None,
            'phi_trajectory': None,
            'theta_trajectory': None,
            'psi_trajectory': None,
            'v_trajectory': None,
            'idx_history': None,
            'obstacles': None
        }

    def savePickle(self, pickle_file:str) -> None:
        """save pickle file"""
        self.pickle_history['x_history'] = self.x_history
        self.pickle_history['y_history'] = self.y_history
        self.pickle_history['z_history'] = self.z_history

        self.pickle_history['phi_history'] = self.phi_history
        self.pickle_history['theta_history'] = self.theta_history
        self.pickle_history['psi_history'] = self.psi_history
        self.pickle_history['v_history'] = self.v_history

        self.pickle_history['x_trajectory'] = self.x_trajectory
        self.pickle_history['y_trajectory'] = self.y_trajectory
        self.pickle_history['z_trajectory'] = self.z_trajectory
        self.pickle_history['phi_trajectory'] = self.phi_trajectory
        self.pickle_history['theta_trajectory'] = self.theta_trajectory
        self.pickle_history['psi_trajectory'] = self.psi_trajectory
        self.pickle_history['v_trajectory'] = self.v_trajectory
        self.pickle_history['idx_history'] = self.idx_history

        #check if config Obstacles is None
        #self.pickle_history['obstacles'] = Config.OBSTACLES

        with open(pickle_file+'.pkl', 'wb') as handle:
            pkl.dump(self.pickle_history, handle)

        print('Saved pickle file: ', pickle_file)

    def positionCallback(self, msg):
        """
        ENU 
        """

        self.state_info[0] = msg.pose.pose.position.x
        self.state_info[1] = msg.pose.pose.position.y
        self.state_info[2] = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_tools.euler_from_quaternion(
            qx, qy, qz, qw)

        self.state_info[3] = roll
        self.state_info[4] = pitch
        self.state_info[5] = yaw

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        
        #get magnitude of velocity
        self.state_info[6] = np.sqrt(vx**2 + vy**2 + vz**2)
        #self.state_info[6] = #msg.twist.twist.linear.x
        self.control_info[0] = msg.twist.twist.angular.x
        self.control_info[1] = msg.twist.twist.angular.y
        self.control_info[2] = msg.twist.twist.angular.z
        self.control_info[3] = msg.twist.twist.linear.x
    
        self.x_history.append(self.state_info[0])
        self.y_history.append(self.state_info[1])
        self.z_history.append(self.state_info[2])
        self.phi_history.append(self.state_info[3])
        self.theta_history.append(self.state_info[4])
        self.psi_history.append(self.state_info[5])
        self.v_history.append(self.state_info[6])

    def computeError(self, current_state:list, desired_state:list) -> list:
        """computeError"""
        error = []
        #catch list index out of range
        for i in range(len(current_state)):
            error.append(desired_state[i] - current_state[i])
        return error
    
    def publishTrajectory(self, traj_dictionary:dict, 
                          state_idx:int, command_idx:int) -> float:
        """
        publishTrajectory
        
        Watch out for the sign convention changes between ENU and NED
        """

        traj_x = np.array(traj_dictionary['x'])
        traj_y = np.array(traj_dictionary['y'])
        traj_z = np.array(traj_dictionary['z'])
        
        traj_phi = np.array(traj_dictionary['phi'])
        traj_theta = -np.array(traj_dictionary['theta'])
        traj_psi = np.array(traj_dictionary['psi'])

        traj_v = np.array(traj_dictionary['v'])

        traj_u_phi = np.array(traj_dictionary['u_phi'])
        traj_u_theta = -np.array(traj_dictionary['u_theta'])
        traj_u_psi = np.array(traj_dictionary['u_psi'])
        traj_v_cmd = np.array(traj_dictionary['v_cmd'])
        
        ned_position = quaternion_tools.convertENUToNEDVector(
            traj_x, traj_y, traj_z)
        
        traj_msg = CtlTraj()
        traj_msg.x = list(ned_position[0][0])
        traj_msg.y = list(ned_position[1][0])
        traj_msg.z = list(ned_position[2][0])
        
        traj_msg.roll = list(traj_phi[0])
        traj_msg.pitch = list(traj_theta[0])
        traj_msg.yaw = list(traj_psi[0])
        
        traj_msg.roll_rate = list(traj_u_phi[0])
        traj_msg.pitch_rate = list(traj_u_theta[0])
        traj_msg.yaw_rate = list(traj_u_psi[0])
        traj_msg.vx = list(traj_v[0])
        
        traj_msg.idx = state_idx

        self.x_trajectory.append(traj_x)
        self.y_trajectory.append(traj_y)
        self.z_trajectory.append(traj_z)
        self.v_trajectory.append(traj_v_cmd)
        self.phi_trajectory.append(traj_phi)
        self.theta_trajectory.append(traj_theta)
        self.psi_trajectory.append(traj_psi)
        self.idx_history.append(state_idx)
        self.traj_pub.publish(traj_msg)

        print("trajectory", traj_phi[0][state_idx])
        return traj_phi[0][state_idx+1]



def initFWMPC() -> AirplaneSimpleModelMPC:

    simple_airplane_model = AirplaneSimpleModel()
    simple_airplane_model.set_state_space()
    
    airplane_params = {
        'u_psi_min': np.deg2rad(-35), #rates
        'u_psi_max': np.deg2rad(35), #
        'u_phi_min': np.deg2rad(-55),
        'u_phi_max': np.deg2rad(55),
        'u_theta_min': np.deg2rad(-10),
        'u_theta_max': np.deg2rad(10),
        'z_min': 35,
        'z_max': 60,    
        'v_cmd_min': 15,
        'v_cmd_max': 23,
        'theta_min': np.deg2rad(-10),
        'theta_max': np.deg2rad(10),
        'phi_min': np.deg2rad(-55),
        'phi_max': np.deg2rad(55),
        'airspeed_min': 15,
        'airspeed_max': 23,
    }


    Q = ca.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R = ca.diag([0.8, 0.8, 0.8, 1.0])

    simple_mpc_fw_params = {
        'model': simple_airplane_model,
        'dt_val': 0.05,
        'N': 20,
        'Q': Q,
        'R': R
    }

    fw_mpc = AirplaneSimpleModelMPC(simple_mpc_fw_params, 
                                    airplane_params)

    return fw_mpc

def main(args=None):
    rclpy.init(args=args)

    control_idx = 10
    state_idx = 5
    idx_buffer = 1

    fw_mpc = initFWMPC()
    mpc_traj_node = MPCTrajFWPublisher()
    
    rclpy.spin_once(mpc_traj_node, timeout_sec=3.0)

    dist_error_tol = 20 
    desired_state_info = mpc_traj_node.get_next_goal_point()

    if desired_state_info == None:
        while desired_state_info == None:
            print("getting waypoints")
            rclpy.spin_once(mpc_traj_node, timeout_sec=3.0)
            desired_state_info = mpc_traj_node.get_next_goal_point()

    desired_phi = mpc_traj_node.state_info[3]

    desired_state = [
            desired_state_info.x, 
            desired_state_info.y,
            desired_state_info.z, 
            desired_state_info.phi, # need to update this 
            desired_state_info.theta, 
            desired_state_info.psi, 
            desired_state_info.v] 

    print("desired state: ", desired_state)

    #get the time to find solution 
    start_time = time.time()
    projected_controls, projected_states = fw_mpc.warmUpSolution(
        mpc_traj_node.state_info,
        desired_state, 
        mpc_traj_node.control_info)

    traj_dictionary = fw_mpc.returnTrajDictionary(
        projected_controls, projected_states)

    # print("traj state: ", traj_state)
    end_time = time.time()

    control_idx = fw_mpc.set_state_control_idx(fw_mpc.mpc_params, 
        end_time - start_time, idx_buffer)

    state_idx = fw_mpc.set_state_control_idx(fw_mpc.mpc_params,
        end_time - start_time, idx_buffer)

    mpc_traj_node.publishTrajectory(
        traj_dictionary, 
        state_idx, 
        control_idx)

    while rclpy.ok():
          
        rclpy.spin_once(mpc_traj_node)
        
        #predict the next state
        offset_state = [mpc_traj_node.state_info[0], 
                        mpc_traj_node.state_info[1], 
                        mpc_traj_node.state_info[2], 
                        mpc_traj_node.state_info[3],  
                        mpc_traj_node.state_info[4], 
                        mpc_traj_node.state_info[5],  
                        mpc_traj_node.state_info[6]] 
        
        if mpc_traj_node.found_new_path is True:
            print("found new path")
            desired_state_info = mpc_traj_node.get_next_goal_point()
            desired_state = [
                desired_state_info.x, 
                desired_state_info.y,
                desired_state_info.z, 
                desired_state_info.phi, 
                desired_state_info.theta, 
                desired_state_info.psi, 
                desired_state_info.v]
            
            # print("desired state: ", desired_state)
            # mpc_traj_node.wp_index += 1

            #mpc_traj_node.destroy_node()
            #rclpy.shutdown()
            #return
            

        fw_mpc.reinitStartGoal(offset_state, desired_state)
        start_time = time.time()
        projected_controls, projected_states = fw_mpc.solveMPCRealTimeStatic(
            offset_state,
            desired_state,
            mpc_traj_node.control_info)
        end_time = time.time()
        print("time to solve: ", end_time - start_time)

        control_idx = fw_mpc.set_state_control_idx(fw_mpc.mpc_params, 
            end_time - start_time, idx_buffer=idx_buffer)

        state_idx = fw_mpc.set_state_control_idx(fw_mpc.mpc_params,
            end_time - start_time, idx_buffer=idx_buffer, use_buffer=True)

        traj_dictionary = fw_mpc.returnTrajDictionary(
            projected_controls, projected_states)

        desired_phi = mpc_traj_node.publishTrajectory(traj_dictionary, 
                                        state_idx, 
                                        control_idx)
        

        distance_error = np.linalg.norm(np.array(desired_state[:2]) - \
                                        np.array(mpc_traj_node.state_info[:2]))

        # distance_error = np.linalg.norm(goal_state_error[0:2])
        print("desired state: ", desired_state[:2])
        print("current state: ", mpc_traj_node.state_info[:2])
        print("distance error: ", distance_error)
        print("\n")

        # update this obstacle to use KNN to find closest  
        # obstacles within the vicinty, need to update the constraints
        if Config.OBSTACLE_AVOID:
            #check if within obstacle SD
            # rclpy.spin_once(mpc_traj_node)

            current_x = mpc_traj_node.state_info[0]
            current_y = mpc_traj_node.state_info[1]

            # if isCollision(current_x, current_y) == True:
            #     print("trajectory x", traj_dictionary['x'])
            #     print("trajectory y", traj_dictionary['y'])
            #     #send 0 velocity command
            #     ref_state_error = [0.0, 0.0, 0.0, 0.0]
            #     ref_control_error = [0.0, 0.0, 0.0, 0.0]

            #     mpc_traj_node.publishTrajectory(traj_dictionary, 
            #                                     state_idx, 
            #                                     control_idx)

                # mpc_traj_node.destroy_node()
                # rclpy.shutdown()
                # return

        if distance_error < dist_error_tol:
            """
            refactor add the length of N of commands to 
            stay where you are at 
            """
            desired_state_info = mpc_traj_node.get_next_goal_point()
            desired_state = [
                desired_state_info.x, 
                desired_state_info.y,
                desired_state_info.z, 
                desired_state_info.phi, 
                desired_state_info.theta, 
                desired_state_info.psi, 
                desired_state_info.v]
            
            print("going to next goal point: ", desired_state)
            mpc_traj_node.wp_index += 1

            #mpc_traj_node.destroy_node()
            #rclpy.shutdown()
            #return 

        # rclpy.spin_once(mpc_traj_node)

if __name__ == '__main__':
    main()
