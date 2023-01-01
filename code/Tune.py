import cv2
import time
from Rover import RoverState


class Tuning:
    # constants
    TUNING_WINDOW = 'Tuning window'

    def init_tuning_window(self, Rover):
        cv2.namedWindow(self.TUNING_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TUNING_WINDOW, 350, 600)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass

        # --------------------------------------create trackbars for perception.---------------------------------------

        cv2.createTrackbar('R', self.TUNING_WINDOW, 0, 255, nothing)
        cv2.setTrackbarPos('R', self.TUNING_WINDOW, Rover.red_threshold)

        cv2.createTrackbar('G', self.TUNING_WINDOW, 0, 255, nothing)
        cv2.setTrackbarPos('G', self.TUNING_WINDOW, Rover.green_threshold)

        cv2.createTrackbar('B', self.TUNING_WINDOW, 0, 255, nothing)
        cv2.setTrackbarPos('B', self.TUNING_WINDOW, Rover.blue_threshold)

        cv2.createTrackbar('Decision_M_S', self.TUNING_WINDOW, 0, 10, nothing)
        cv2.setTrackbarPos('Decision_M_S', self.TUNING_WINDOW, Rover.decision_mask_size)

        cv2.createTrackbar('Mapping_M_S', self.TUNING_WINDOW, 0, 10, nothing)
        cv2.setTrackbarPos('Mapping_M_S', self.TUNING_WINDOW, Rover.mapping_mask_size)

        cv2.createTrackbar('Steer_limit_update', self.TUNING_WINDOW, 0, 15, nothing)
        cv2.setTrackbarPos('Steer_limit_update', self.TUNING_WINDOW, Rover.steer_update_limit)

        cv2.createTrackbar('Pitch_limit_update', self.TUNING_WINDOW, 0, 5, nothing)
        cv2.setTrackbarPos('Pitch_limit_update', self.TUNING_WINDOW, Rover.pitch_update_limit)

        cv2.createTrackbar('Roll_limit_update', self.TUNING_WINDOW, 0, 5, nothing)
        cv2.setTrackbarPos('Roll_limit_update', self.TUNING_WINDOW, Rover.roll_update_limit)

        # ---------------------------------------create trackbars for decision.-----------------------------------------

        cv2.createTrackbar('Max_rock_dist', self.TUNING_WINDOW, 0, 150, nothing)
        cv2.setTrackbarPos('Max_rock_dist', self.TUNING_WINDOW, Rover.max_rock_distance)

        cv2.createTrackbar('Max_rock_angle', self.TUNING_WINDOW, 0, 30, nothing)
        cv2.setTrackbarPos('Max_rock_angle', self.TUNING_WINDOW, int(15 + Rover.max_rock_angle))

        cv2.createTrackbar('Rock_time_max', self.TUNING_WINDOW, 0, 25, nothing)
        cv2.setTrackbarPos('Rock_time_max', self.TUNING_WINDOW, Rover.rock_time_max)

        cv2.createTrackbar('Rock_stuck_time_max', self.TUNING_WINDOW, 0, 25, nothing)
        cv2.setTrackbarPos('Rock_stuck_time_max', self.TUNING_WINDOW, Rover.rock_stuck_time_max)

        cv2.createTrackbar('Stuck_time_max', self.TUNING_WINDOW, 0, 10, nothing)
        cv2.setTrackbarPos('Stuck_time_max', self.TUNING_WINDOW, Rover.stuck_time_max)

        cv2.createTrackbar('Wall_offset_weight', self.TUNING_WINDOW, 0, 10, nothing)
        cv2.setTrackbarPos('Wall_offset_weight', self.TUNING_WINDOW, int(Rover.offset_weight*10))

        cv2.createTrackbar('Go_forward', self.TUNING_WINDOW, 0, 500, nothing)
        cv2.setTrackbarPos('Go_forward', self.TUNING_WINDOW, Rover.go_forward)

        cv2.createTrackbar('Stop_forward', self.TUNING_WINDOW, 0, 250, nothing)
        cv2.setTrackbarPos('Stop_forward', self.TUNING_WINDOW, Rover.stop_forward)

        cv2.createTrackbar('Max_velocity', self.TUNING_WINDOW, 0, 5, nothing)
        cv2.setTrackbarPos('Max_velocity', self.TUNING_WINDOW, Rover.max_vel)

        cv2.createTrackbar('Throttle_set', self.TUNING_WINDOW, 0, 10, nothing)
        cv2.setTrackbarPos('Throttle_set', self.TUNING_WINDOW, int(Rover.throttle_set*10))

        cv2.createTrackbar('Brake_set', self.TUNING_WINDOW, 0, 20, nothing)
        cv2.setTrackbarPos('Brake_set', self.TUNING_WINDOW, Rover.brake_set)

        cv2.createTrackbar('wall_side', self.TUNING_WINDOW, 0, 1, nothing)
        cv2.setTrackbarPos('wall_side', self.TUNING_WINDOW, Rover.wall_side)

        cv2.waitKey(1)

    def get_tunning_paramesters(self):
        # Get current positions of all trackbars
        Rover = RoverState()

        # -----------------------------------------------------Get perception-------------------------------------------
        Rover.red_threshold = cv2.getTrackbarPos('R', self.TUNING_WINDOW)
        Rover.green_threshold = cv2.getTrackbarPos('G', self.TUNING_WINDOW)
        Rover.blue_threshold = cv2.getTrackbarPos('B', self.TUNING_WINDOW)

        Rover.decision_mask_size = cv2.getTrackbarPos('Decision_M_S', self.TUNING_WINDOW)
        Rover.mapping_mask_size = cv2.getTrackbarPos('Mapping_M_S', self.TUNING_WINDOW)

        Rover.steer_update_limit = cv2.getTrackbarPos('Steer_limit_update', self.TUNING_WINDOW)
        Rover.pitch_update_limit = cv2.getTrackbarPos('Pitch_limit_update', self.TUNING_WINDOW)
        Rover.roll_update_limit = cv2.getTrackbarPos('Roll_limit_update', self.TUNING_WINDOW)

        # -------------------------------------------------------Get decision-------------------------------------------
        Rover.max_rock_distance = cv2.getTrackbarPos('Max_rock_dist', self.TUNING_WINDOW)
        Rover.max_rock_angle = cv2.getTrackbarPos('Max_rock_angle', self.TUNING_WINDOW)-15

        Rover.rock_time_max = cv2.getTrackbarPos('Rock_time_max', self.TUNING_WINDOW)
        Rover.rock_stuck_time_max = cv2.getTrackbarPos('Rock_stuck_time_max', self.TUNING_WINDOW)
        Rover.stuck_time_max = cv2.getTrackbarPos('Stuck_time_max', self.TUNING_WINDOW)

        Rover.offset_weight = cv2.getTrackbarPos('Wall_offset_weight', self.TUNING_WINDOW) / 10
        Rover.go_forward = cv2.getTrackbarPos('Go_forward', self.TUNING_WINDOW)
        Rover.stop_forward = cv2.getTrackbarPos('Stop_forward', self.TUNING_WINDOW)
        Rover.max_vel = cv2.getTrackbarPos('Max_velocity', self.TUNING_WINDOW)
        Rover.throttle_set = cv2.getTrackbarPos('Throttle_set', self.TUNING_WINDOW) / 10
        Rover.brake_set = cv2.getTrackbarPos('Brake_set', self.TUNING_WINDOW)
        if cv2.getTrackbarPos('wall_side', self.TUNING_WINDOW) == 1:
            Rover.wall_side = 1
        elif cv2.getTrackbarPos('wall_side', self.TUNING_WINDOW) == 0:
            Rover.wall_side = -1

        return Rover


if __name__ == '__main__':
    test = Tuning()
    test.init_tuning_window()
    cv2.waitKey(10)
    #time.sleep(3)
    print(test.get_tunning_paramesters().max_rock_angle)
