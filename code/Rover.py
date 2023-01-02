import matplotlib.image as mpimg
import numpy as np

# Read in ground truth map and create 3-channel green version for overplotting
# NOTE: images are read in by default with the origin (0, 0) in the upper left
# and y-axis increasing downward.
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
# This next line creates arrays of zeros in the red and blue channels
# and puts the map into the green channel.  This is why the underlying
# map output looks green in the display image
ground_truth_3d = np.dstack((ground_truth * 0, ground_truth * 255, ground_truth * 0)).astype(float)


class RoverState():
    def __init__(self):
        self.start_time = None  # To record the start time of navigation
        self.total_time = None  # To record total duration of naviagation
        self.stuck_time = 0  # To record moment that got stuck
        self.stuck_time_max = 5  # ------------------------------> hyperparameter to be manipulated later
        self.rock_time = 0  # To record moment that started to go for the near rock sample
        self.rock_time_max = 20  # ------------------------------> hyperparameter to be manipulated later
        self.rock_stuck_time_max = 20  # ------------------------------> hyperparameter to be manipulated later
        self.img = None  # Current camera image
        self.pos = None  # Current position (x, y)
        self.yaw = None  # Current yaw angle
        self.pitch = None  # Current pitch angle
        self.roll = None  # Current roll angle
        self.vel = None  # Current velocity
        self.steer = 0  # Current steering angle
        self.throttle = 0  # Current throttle value
        self.brake = 0  # Current brake value
        self.nav_angles = None  # Angles of navigable terrain pixels
        self.nav_dists = None  # Distances of navigable terrain pixels
        self.samples_angles = None  # Angles of rock sample pixels
        self.samples_dists = None  # Distances of rock sample pixels
        self.ground_truth = ground_truth_3d  # Ground truth worldmap
        self.mode = ['forward']  # Current mode (can be forward or stop)
        self.throttle_set = 0.5  # Throttle setting when accelerating
        self.brake_set = 10  # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 50  # Threshold to initiate stopping
        self.go_forward = 250  # Threshold to go forward again
        self.max_vel = 2  # Maximum velocity (meters/second)
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=float)
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=float)
        self.samples_pos = None  # To store the actual sample positions
        self.samples_to_find = 0  # To store the initial count of samples
        self.samples_located = 0  # To store number of samples located on map
        self.samples_collected = 0  # To count the number of samples collected
        self.near_sample = 0  # Will be set to telemetry value data["near_sample"]
        self.picking_up = 0  # Will be set to telemetry value data["picking_up"]
        self.send_pickup = False  # Set to True to trigger rock pickup

        self.wall_side = 1  # 1 for right wall -1 for left wall
        self.offset_weight = 0.8  # weight of the std to be added to the nav direction -> hyperparameter to be manipulated later
        self.approch_rock_throttle = 0.5
        # Rock Max distance and angle
        self.max_rock_distance = 50
        self.max_rock_angle = 20  # --> in degree

        # Perception file
        # Threshold for the navigable terrain
        self.red_threshold = 180  #
        self.green_threshold = 180  #
        self.blue_threshold = 160
        # limiting the view of the rover
        self.decision_mask_size = 8  #
        self.mapping_mask_size = 6
        # set limits for pitch roll and steering for update
        self.steer_update_limit = 5
        self.pitch_update_limit = 1
        self.roll_update_limit = 1
