import numpy as np
import cv2

show_pipeline = True
dst_size = 5


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(170, 170, 140)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
                   & (img[:, :, 1] > rgb_thresh[1]) \
                   & (img[:, :, 2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


# Define a function that applies hysteresis thresholding
def hysteresis_threshold(image, high_threshold=(160, 160, 160), low_threshold=(80, 80, 80)):
    # step1 classify pixels to three sets
    M, N, _ = image.shape
    threshed = np.zeros_like(image[:, :, 0])

    # strong ground pixels --> terrain for sure
    strong_threshed = (image[:, :, 0] > high_threshold[0]) \
                      & (image[:, :, 1] > high_threshold[1]) \
                      & (image[:, :, 2] > high_threshold[2])

    # obstacles
    zero_threshed = (image[:, :, 0] < low_threshold[0]) \
                    & (image[:, :, 1] < low_threshold[1]) \
                    & (image[:, :, 2] < low_threshold[2])

    # weak edges
    weak_threshed = (image[:, :, 0] > low_threshold[0]) & (image[:, :, 0] < high_threshold[0]) \
                    & (image[:, :, 1] > low_threshold[1]) & (image[:, :, 1] < high_threshold[1]) \
                    & (image[:, :, 2] > low_threshold[2]) & (image[:, :, 2] < high_threshold[2])

    # Set same intensity value for all edge pixels
    threshed[strong_threshed] = 1
    threshed[zero_threshed] = 0
    threshed[weak_threshed] = 127

    # step 2 match weak pixels to high if any of its neighbours is high
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if threshed[i, j] == 127:
                if 255 in [threshed[i + 1, j - 1], threshed[i + 1, j], threshed[i + 1, j + 1], threshed[i, j - 1],
                           threshed[i, j + 1], threshed[i - 1, j - 1], threshed[i - 1, j], threshed[i - 1, j + 1]]:
                    threshed[i, j] = 1
                else:
                    threshed[i, j] = 0
    return threshed


# Define a function that calls OTSU thresholding on the given image
def otsu_thresholding(image, blurring_size=(7, 7)):
    # convert the colored image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurr the image
    blurred = cv2.GaussianBlur(gray, blurring_size, 0)
    # apply OTSU's thresholding
    otsu_threshold, otsu_threshed_image = cv2.threshold(blurred, 0, 1, cv2.THRESH_OTSU)

    return otsu_threshed_image


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    ''' To improve the mapping precision we create this mask so that after the perspective transform and color thresh
        we could deduce the obstacles by subtracting the threshed from this mask'''
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))

    return warped, mask


# Define a function that thresholds the image to find the rocks
def find_rocks(img, thresh=(110, 110, 50)):
    rock_pixels = ((img[:, :, 0] > thresh[0]) \
                   & (img[:, :, 1] > thresh[1]) \
                   & (img[:, :, 2] < thresh[2]))

    colored_pixels = np.zeros_like(img[:, :, 0])
    colored_pixels[rock_pixels] = 1

    return colored_pixels


# Define a function that clips the incoming image so as to overcome camera errors for far objects
# NOTE: the clipping here is in the form of a semicircle
def limit_view(x_pixels, y_pixels, range=(8) * 2 * dst_size):
    distance = np.sqrt(x_pixels ** 2, y_pixels ** 2)
    return x_pixels[distance < range], y_pixels[distance < range]


# Define original clipping functions
def trim_ellipse(image, offset, limit):
    # create a mask image of the same shape as input image, filled with 0s (black color)
    mask = np.zeros_like(image)
    rows, cols = mask.shape
    # create a white filled ellipse  ----------> 3/4
    mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows - offset)), axes=(limit, limit), angle=0,
                       startAngle=180, endAngle=360, color=(255, 255, 255), thickness=-1)
    # Bitwise AND operation to black out regions outside the mask
    return image * mask


def trim_rectangle(image):
    # create a mask image of the same shape as input image, filled with 0s (black color)
    mask = np.zeros_like(image)
    rows, cols, _ = mask.shape
    trimed_perc = 0.5
    start = (0, int(rows * trimed_perc))
    end = (int(cols), int(rows))
    # create a white filled rectangle
    mask = cv2.rectangle(mask, start, end, color=(255, 255, 255), thickness=-1)
    # Bitwise AND operation to black out regions outside the mask
    return np.bitwise_and(image, mask)


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    image = image_original = Rover.img
    # image_blur = cv2.GaussianBlur(image_original, (5, 5), 0)
    # kernel = np.ones((5, 5), np.float32) / 25
    # image_blur = cv2.filter2D(image_original, -1, kernel)
    # image = trim_ellipse(image_blur)

    # Constraint the world map update based on an accepted range of the pitch and roll
    pitch_condition = (abs(Rover.pitch) < 1) or (abs(Rover.pitch - 360) < 1)
    roll_condition = (abs(Rover.roll) < 1) or (abs(Rover.roll - 360) < 1)
    steering_condition = (abs(Rover.steer) < 6) and (abs(Rover.vel) < 2)
    break_condition = Rover.brake == 0
    condition_to_update_worldmap = pitch_condition and roll_condition and steering_condition

    # ------------ 1) Define source and destination points for perspective transform------------------------------------

    # The destination box will be 2*dst_size on each side --> (dont forget to scale back when mapping to world map)

    bottom_offset = 3
    source = np.float32([[14, 140], [300, 140], [200, 95], [120, 95]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])

    # ------------------------------- 2) Apply perspective transform----------------------------------------------------
    warped, obstacle_mask = perspect_transform(image, source, destination)

    # ---------------- 3) Apply color threshold to identify navigable terrain/obstacles/rock samples--------------------
    # threshed = hysteresis_threshold(warped, (170, 170, 170), (100, 100, 100))
    threshed = otsu_thresholding(warped)
    obstacle_map = np.absolute(np.float32(threshed) - 1) * obstacle_mask
    rock_map = find_rocks(warped, (110, 110, 50))

    # Better performance discovered
    '''
    # Clip the upper 40% of the image as the camera performance deteriorates for long distances
    percentage_of_clipping = 0.0
    threshed[0: int(threshed.shape[0] * percentage_of_clipping), :] = 0
    obstacle_map[0: int(obstacle_map.shape[0] * percentage_of_clipping), :] = 0
    '''
    # trimming the perspective transform to provide better mapping fidelity

    # Trimming the decision view window to 8 pixels ahead
    threshed_decision = trim_ellipse(threshed, bottom_offset, 8 * (2 * dst_size))
    # Trimming the mapping view window to 4 pixels ahead for better mapping
    threshed_mapping = trim_ellipse(threshed, bottom_offset, 6 * (2 * dst_size))
    obstacle_map_mapping = trim_ellipse(obstacle_map, bottom_offset, 6 * (2 * dst_size))

    # -------------------- 4) Update Rover.vision_image (this will be displayed on left side of screen)-----------------
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    Rover.vision_image[:, :, 0] = obstacle_map_mapping
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    if rock_map.any():
        Rover.vision_image[:, :, 1] = rock_map * 255
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:, :, 2] = threshed_mapping

    # --------------------------------- 5) Convert map image pixel values to rover-centric coords-----------------------
    # For decision
    xpix_decision, ypix_decision = rover_coords(threshed_decision)
    # For mapping
    xpix, ypix = rover_coords(threshed_mapping)
    obs_xpix, obs_ypix = rover_coords(obstacle_map_mapping)
    rock_xpix, rock_ypix = rover_coords(rock_map)

    # --------------------------------- 6) Convert rover-centric pixel values to world coordinates----------------------
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size

    # xpix_new, ypix_new = limit_view(xpix, ypix)
    # obs_xpix_new, obs_ypix_new = limit_view(obs_xpix, obs_ypix)

    # get the actual terrain mapping
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    # get the actual obstacles mapping
    obs_xpix_world, obs_ypix_world = pix_to_world(obs_xpix, obs_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size,
                                                  scale)
    # get the actual rocks mapping
    rock_xpix_world, rock_ypix_world = pix_to_world(rock_xpix, rock_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw,
                                                    world_size, scale)

    # ---- 7) Update Rover worldmap (to be displayed on right side of screen) --> fidelity is measured from this map----
    if condition_to_update_worldmap:
        # in walkthrough increment 10 but I will put it 255 (max)

        # Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[obs_ypix_world, obs_xpix_world, 0] = 255  # populate the red channel with obstacles
        # Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[rock_ypix_world, rock_xpix_world, 1] = 255  # populate green with rocks
        # Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        Rover.worldmap[y_world, x_world, 2] = 255  # populate the blue channel with navigable terrain

        # remove overlap measurements by giving the upper hand to the terrain  --> sort of improving fidelity
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
        # remove overlap measurements by giving the upper hand to the rock  --> sort of improving fidelity
        rock_pix = Rover.worldmap[:, :, 1] > 0
        Rover.worldmap[rock_pix, 0] = 0

    # ------------------------------8) Convert rover-centric pixel positions to polar coordinates-----------------------

    dist, angles = to_polar_coords(xpix_decision, ypix_decision)
    rock_dist, rock_angle = to_polar_coords(rock_xpix, rock_ypix)
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_dists = dist
    # Rover.nav_angles = rover_centric_angles
    Rover.nav_angles = angles

    # Now we update the two added variables to the rover for rock picking
    '''
    Rover.rock_distance_from_rover = rock_dist
    Rover.rock_angle = rock_angle
    '''
    Rover.samples_dists = rock_dist
    Rover.samples_angles = rock_angle

    # images we want to stream to the debugging mode:
    # image, warped, threshed,  obstacle map, rock map

    if show_pipeline:
        original_image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        cv2.imshow('Original', original_image)

        '''
        blur_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2RGB)
        cv2.imshow('Original Blurred', blur_image)
        '''
        '''
        image_bgt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Limited view', image_bgt)
        '''
        warped_bgt = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        cv2.imshow('Perspective Transform', warped_bgt)

        threshed = threshed * 255
        cv2.imshow('Threshed terrain', threshed)

        obstacle_map = obstacle_map * 255
        cv2.imshow('Obstacle', obstacle_map)

        rock_map = rock_map * 255
        cv2.imshow('Rock', rock_map)

        # Terrain for decision and for mapping
        threshed_decision = threshed_decision
        cv2.imshow('Threshed terrain decision', threshed_decision)

        threshed_mapping = threshed_mapping
        cv2.imshow('Threshed terrain mapping', threshed_mapping)

        # obstacles for mapping
        obstacle_map_mapping = obstacle_map_mapping
        cv2.imshow('obstacle mapping', obstacle_map_mapping)

        '''
        # still need to debug the rover coordinates map and the real world map
        world_new = np.zeros((300, 300))
        world_new[ypix.astype(int), xpix.astype(int)] = 255
        cv2.imshow("Rover Coordinates", world_new)
        '''
        mean_dir = np.mean(angles)
        mean_dir2 = np.mean(rock_angle)
        arrow_length = np.mean(dist)
        rock_arrow_length = np.mean(rock_dist)
        x_arrow = arrow_length * np.cos(mean_dir)
        y_arrow = arrow_length * np.sin(mean_dir)
        x_rock = rock_arrow_length * np.cos(mean_dir2)
        y_rock = rock_arrow_length * np.sin(mean_dir2)
        if ((x_arrow == x_arrow) and (y_arrow == y_arrow)):
            color = (0, 0, 255)
            thickness = 2
            view = cv2.rotate(threshed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
            start_point = (int(view.shape[1]), int(view.shape[0] / 2))
            end_point = (int(x_arrow), int(y_arrow) + int(view.shape[0] / 2))
            direction = cv2.arrowedLine(view, start_point, end_point, color, thickness)
            direction = cv2.rotate(direction, cv2.ROTATE_180)
            cv2.imshow('Nav Direction', direction)

        if ((x_rock == x_rock) and (y_rock == y_rock)):
            color2 = (255, 0, 0)
            thickness = 2
            view = cv2.rotate(rock_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
            start_point = (int(view.shape[1]), int(view.shape[0] / 2))
            end_point = (int(x_rock), int(y_rock) + int(view.shape[0] / 2))
            direction = cv2.arrowedLine(view, start_point, end_point, color2, thickness)
            direction = cv2.rotate(direction, cv2.ROTATE_180)
            cv2.imshow('Nav Direction in case of rock', direction)

        cv2.waitKey(5)

    return Rover
