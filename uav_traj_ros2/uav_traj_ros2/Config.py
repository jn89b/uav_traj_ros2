# Change this to a yaml file 
import numpy as np

# Since our simulation is huge, lets scale it down
SCALE_SIM = 25

## START 
START_X = -250
START_Y = -150
START_PSI = np.deg2rad(0)

## GOAL
GOAL_X = 1500
GOAL_Y = 1500
GOAL_Z = 45
GOAL_PSI = 0

#### OBSTACLES ####
OBSTACLE_AVOID = False
MOVING_OBSTACLE = False
MULTIPLE_OBSTACLE_AVOID = False
ROBOT_DIAMETER = 5

RADAR_AVOID = False
RADAR_USE_CASADI = False

OBSTACLE_X = 0
OBSTACLE_Y = 0         
OBSTACLE_DIAMETER = 75
OBSTACLE_VX = 0.0
OBSTACLE_VY = 0.0

X_MAX = 150
Y_MAX = 150
X_MIN = -150
Y_MIN = 0


#NLP solver options
MAX_ITER = 100
MAX_TIME = 0.1
PRINT_LEVEL = 2
ACCEPT_TOL = 1e-2
ACCEPT_OBJ_TOL = 1e-2   
PRINT_TIME = 0


#Target options
TARGET_DISCHARGE_RATE = 0.1
