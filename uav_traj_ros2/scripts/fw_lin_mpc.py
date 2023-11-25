#!/usr/bin/env python3
import casadi as ca
import rclpy 
import numpy as np
import pandas as pd
import pickle as pkl

#from drone_interfaces.msg import Telem, CtlTraj
from drone_interfaces.msg import Telem, RCCmd

from uav_traj_ros2.src.aircraft.AircraftDynamics import LinearizedAircraftCasadi
from uav_traj_ros2.src.mpc import FixedWingMPC 
from uav_traj_ros2.src.mpc.FixedWingMPC import LinearizedAircraftMPC

from uav_traj_ros2.src.math_lib.VectorOperations import euler_dcm_body_to_inertial, euler_dcm_inertial_to_body

from rclpy.node import Node
from uav_traj_ros2.src.Utils import get_airplane_params
import mavros
from mavros.base import SENSOR_QOS


class FWLinMPC(Node):
    def __init__(self, mpc_constraints:dict):
        super().__init__('fw_lin_mpc')        
        self.get_logger().info('Starting MPC Traj FW Publisher')
        
        self.mpc_constraints = mpc_constraints
        
        self.state_info = [
            0.0, #x
            0.0, #y
            0.0, #z
            0.0, #u
            0.0, #v
            0.0, #w
            0.0, #phi
            0.0, #theta
            0.0, #psi
            0.0, #p
            0.0, #q
            0.0, #r
        ]
        
        self.control_info = [
            0.0, #delta_e
            0.0, #delta_t
            0.0, #delta_a
            0.0, #delta_r
        ]
        

        self.state_sub = self.create_subscription(Telem, 
                                                  'telem', 
                                                  self.stateCallback, 
                                                  1)
        
        self.control_sub = self.create_subscription(RCCmd, 
                                                    'rc_cmd', 
                                                    self.controlCallback, 
                                                    1)

        self.initHistory()
        
        
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

    def stateCallback(self, msg:Telem) -> None:
        self.state_info[0] = msg.x
        self.state_info[1] = msg.y
        self.state_info[2] = msg.z
        vx = msg.vx
        vy = msg.vy
        vz = msg.vz
        
        self.state_info[6] = msg.roll
        self.state_info[7] = msg.pitch
        self.state_info[8] = msg.yaw
        self.state_info[9] = msg.roll_rate
        self.state_info[10] = msg.pitch_rate
        self.state_info[11] = msg.yaw_rate
        
        
        #convert velocities to body frame
        R = euler_dcm_inertial_to_body(self.state_info[6],
                                       self.state_info[7],
                                       self.state_info[8])
        
        v_inertial = np.array([vx, vy, vz])
        v_body = np.matmul(R, v_inertial)
        
        self.state_info[3] = v_body[0]
        self.state_info[4] = v_body[1]
        self.state_info[5] = v_body[2]
        
    def convertPWMtoActual(self, pwm_control:np.ndarray) -> np.ndarray:
        """
        This sets a scale factor for the RC values to convert
        from PWM commands to actual values
        Assumption is the PWM values range from 1000 to 2000
        
        Equation is as follows:
        actual_value = min_actual + ((pwm_value - min_pwm) * (range_actual / range_pwm))
        
        Aileron in radians
        Elevator in radians
        Rudder in radians
        Throttle in percentage of max thrust
        """
        mpc_constraints = self.mpc_constraints
        
        aileron_range_rad = mpc_constraints['delta_a_max'] - mpc_constraints['delta_a_min']
        elevator_range_rad = mpc_constraints['delta_e_max'] - mpc_constraints['delta_e_min']
        rudder_range_rad = mpc_constraints['delta_r_max'] - mpc_constraints['delta_r_min']
        #thrust_range = mpc_constraints['delta_t_max'] - mpc_constraints['delta_t_min']
        thrust_range = 1.0 
        
        min_rc = 1000
        max_rc = 2000
        rc_range = max_rc - min_rc
        
        control_info = np.zeros(4)
        
        print("aileron deg range: {}".format(np.rad2deg(aileron_range_rad)))
        control_info[0] = mpc_constraints['delta_a_min'] + \
            ((pwm_control[0] - min_rc) * (aileron_range_rad / rc_range))
        
        control_info[1] = mpc_constraints['delta_e_min'] + \
            ((pwm_control[1] - min_rc) * (elevator_range_rad / rc_range))
            
        control_info[2] = mpc_constraints['delta_r_min'] + \
            ((pwm_control[2] - min_rc) * (rudder_range_rad / rc_range))
            
        control_info[3] = mpc_constraints['delta_t_min'] + \
            ((pwm_control[3] - min_rc) * (thrust_range / rc_range))
            
        return control_info
    
    def controlCallback(self, msg:RCCmd) -> None:
        """
        These value range from 1000 to 2000
        Need to map this back to actual values
        """
        pwm_control = np.zeros(4)
        pwm_control[0]= msg.aileron
        pwm_control[1]= msg.elevator
        pwm_control[2]= msg.rudder
        pwm_control[3]= msg.thrust
        
        # print("PWM Control: {}".format(pwm_control))
        
        self.control_info = self.convertPWMtoActual(pwm_control)
        #print("Control Info: {}".format(self.control_info))             
        deg_elev = np.rad2deg(self.control_info[1])
        deg_ail = np.rad2deg(self.control_info[0])
        deg_rud = np.rad2deg(self.control_info[2])
        
        # print("Elev: {}, Ail: {}, Rud: {}".format(deg_elev, deg_ail, deg_rud))

def main(args=None):
    
    aircraft_csv = "/home/justin/ros2_ws/src/uav_traj_ros2/uav_traj_ros2/uav_traj_ros2/aircraft_data/SIM_Plane_h_vals.csv"
    aircraft_df = pd.read_csv(aircraft_csv)
    aircraft_params = get_airplane_params(aircraft_df)
    
    A_full_pickle = "/home/justin/ros2_ws/src/uav_traj_ros2/uav_traj_ros2/uav_traj_ros2/aircraft_data/A_full.pkl"
    B_full_pickle = "/home/justin/ros2_ws/src/uav_traj_ros2/uav_traj_ros2/uav_traj_ros2/aircraft_data/B_full.pkl"
    
    with open(A_full_pickle, 'rb') as file:
        A_full = pkl.load(file)
        
    with open(B_full_pickle, 'rb') as file:
        B_full = pkl.load(file)
        
    lin_aircraft_ca = LinearizedAircraftCasadi(aircraft_params, 
                                               A_full,
                                               B_full)
    
    
    #terminal conditions
    goal_u = 15.0
    goal_w = 0.0
    goal_q = 0.0
    goal_theta = np.deg2rad(-0.03)
    goal_h = 1.0
    goal_x = 250
    goal_v = 0.0
    goal_p = 0.0
    goal_r = 0.0
    goal_phi = np.deg2rad(0.0)
    goal_psi = np.deg2rad(30.0)
    goal_y = 0.0

    #weighting matrices for state
    Q = np.diag([
        1.0, #u
        0.0, #w
        0.0, #q
        0.0, #theta
        1.0, #h
        0.0, #x
        0.0, #v
        0.0, #p
        0.0, #r
        0.0, #phi
        1.0, #psi
        0.0, #y
    ])

    R = np.diag([
        0.0, #delta_e
        0.0, #delta_t
        0.0, #delta_a
        0.0, #delta_r
    ])

    mpc_params = {
        'model': lin_aircraft_ca,
        'dt_val': 0.1,
        'N': 15,
        'Q': Q,
        'R': R,
    }

    lin_mpc_constraints = {
        'delta_e_min': np.deg2rad(-30),
        'delta_e_max': np.deg2rad(30),
        'delta_t_min': 0.05,
        'delta_t_max': 0.75,
        'delta_a_min': np.deg2rad(-25),
        'delta_a_max': np.deg2rad(25),
        'delta_r_min': np.deg2rad(-30),
        'delta_r_max': np.deg2rad(30),
        'u_min': 15,
        'u_max': 35,
        'w_min': -0.5,
        'w_max': 0.5,
        'q_min': np.deg2rad(-60),
        'q_max': np.deg2rad(60),
        'theta_min': np.deg2rad(-35),
        'theta_max': np.deg2rad(35),
        'v_min': -35, #don't need this really
        'v_max': 35,  #don't need this really
        'p_min': np.deg2rad(-60),
        'p_max': np.deg2rad(60),
        'r_min': np.deg2rad(-60),
        'r_max': np.deg2rad(60),
        'phi_min': np.deg2rad(-60),
        'phi_max': np.deg2rad(60)
    }

    states = {
        'u': 25,
        'w': 0.0,
        'q': 0,
        'theta': np.deg2rad(-0.03),
        'h': 0.0,
        'x': 0.0,
        'v': 0.0,
        'p': 0.0,
        'r': 0.0,
        'phi': 0.0,
        'psi': 0.0,
        'y': 0.0,
    }

    controls = {
        'delta_e': np.deg2rad(0),
        'delta_t': 0.1,
        'delta_a': np.deg2rad(0),
        'delta_r': np.deg2rad(0),
    }

    start_state = np.array([states['u'],
                            states['w'],
                            states['q'],
                            states['theta'],
                            states['h'],
                            states['x'],
                            states['v'],
                            states['p'],
                            states['r'],
                            states['phi'],
                            states['psi'],
                            states['y']])

    start_control = np.array([controls['delta_e'],
                            controls['delta_t'],
                            controls['delta_a'],
                            controls['delta_r']])


    goal_state = np.array([goal_u,
                            goal_w,
                            goal_q,
                            goal_theta,
                            goal_h,
                            goal_x,
                            goal_v,
                            goal_p,
                            goal_r,
                            goal_phi,
                            goal_psi,
                            goal_y])

    #begin mpc
    lin_mpc = LinearizedAircraftMPC(mpc_params, lin_mpc_constraints)

    lin_mpc.initDecisionVariables()
    lin_mpc.reinitStartGoal(start_state, goal_state)
    lin_mpc.computeCost()
    lin_mpc.defineBoundaryConstraints()
    lin_mpc.addAdditionalConstraints()

    control_results, state_results = lin_mpc.solveMPCRealTimeStatic(
        start_state, goal_state, start_control)

    #unpack the results
    control_results = lin_mpc.unpack_controls(control_results)
    state_results = lin_mpc.unpack_states(state_results)

    # begin node
    rclpy.init(args=args)
    
    control_idx = 1
    state_idx = 1
    
    fw_mpc = FWLinMPC(lin_mpc_constraints)

    rclpy.spin_once(fw_mpc, timeout_sec=0.001)

    while rclpy.ok():
        rclpy.spin_once(fw_mpc, timeout_sec=1/100)        

if __name__ == '__main__':
    main()
    