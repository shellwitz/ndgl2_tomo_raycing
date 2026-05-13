from dataclasses import dataclass

import numpy as np

RAY_SOURCE_DIST = 2
RAY_ANGLE = np.pi/4
BEAM_NUM = 5
GRID_SIZE = 10


def ray_aa_bb(aa, bb, r_origin, r_dir): #expect shape (2,) np arrays, aa lower left corner bb upper right

    t_latest_entry = 0.0
    t_earliest_exit = np.inf

    for axis in range(2):
        if r_dir[axis] == 0:
            
            if r_origin[axis] < aa[axis] or r_origin[axis] > bb[axis]:
                return -np.inf, -np.inf
            continue
        
        t_near = (aa[axis]-r_origin[axis])/r_dir[axis]

        t_far = (bb[axis]-r_origin[axis])/r_dir[axis]

        if t_near > t_far:
            t_near, t_far = t_far, t_near

        t_latest_entry = max(t_near, t_latest_entry)

        t_earliest_exit = min(t_far, t_earliest_exit)

        if t_latest_entry >= t_earliest_exit:
            #no intersection, >= for not having to deal with tangent lines that cross no area
            return -np.inf, -np.inf
    
    return t_latest_entry, t_earliest_exit

@dataclass
class TraversedPixel:
    i: int
    j: int
    length: float

def amanatides_and_woos_traversal(first_hit_coord, ray_dir, aa, bb):
    step = np.sign(ray_dir)
    step_x = int(step[0])
    step_y = int(step[1])

    pixel_length = 1/GRID_SIZE #length of square is 1

    delta_t = np.abs(pixel_length/ray_dir)
    delta_t_x = delta_t[0]
    delta_t_y = delta_t[1]

    num_pixels_till_coord = (first_hit_coord - aa)/pixel_length

    current_grid_i = int(np.clip(GRID_SIZE-np.ceil(num_pixels_till_coord[1]), 0, GRID_SIZE-1)) #the grid i, j coordinate should be normal matrix indices. Upper left should be 0,0
    current_grid_j = int(np.clip(np.floor(num_pixels_till_coord[0]), 0, GRID_SIZE-1)) #the np.clip is like fancy if checking whether the value is GRID_SIZE and subtracting one if not


    if step_x > 0:
        upper_x_bound = aa[0] + (current_grid_j+1)*pixel_length
        t_x = (upper_x_bound-first_hit_coord[0])/ray_dir[0]
    elif step_x < 0:
        lower_x_bound = aa[0]  + current_grid_j*pixel_length
        t_x = (lower_x_bound-first_hit_coord[0])/ray_dir[0]
    else:
        t_x = np.inf

    
    if step_y > 0:
        upper_y_bound = bb[1] - current_grid_i*pixel_length
        t_y = (upper_y_bound-first_hit_coord[1])/ray_dir[1]
    elif step_y < 0:
        lower_y_bound = bb[1] - (current_grid_i+1)*pixel_length    
        t_y = (lower_y_bound-first_hit_coord[1])/ray_dir[1]
    else:
        t_y = np.inf

    
    traversed_pixels = []
    
    prev_t = 0

    while True:
        if t_x < t_y:

            len_in_cell = np.linalg.norm((t_x - prev_t)*ray_dir)

            traversed_pixels.append(TraversedPixel(current_grid_i, current_grid_j, len_in_cell))

            current_grid_j += step_x

            if current_grid_j < 0 or current_grid_j >= GRID_SIZE:
                break

            prev_t = t_x
            t_x += delta_t_x
            
        else:
            len_in_cell = np.linalg.norm((t_y - prev_t)*ray_dir)
            traversed_pixels.append(TraversedPixel(current_grid_i, current_grid_j, len_in_cell))

            current_grid_i -= step_y

            if current_grid_i < 0 or current_grid_i >= GRID_SIZE:
                break

            prev_t = t_y
            t_y+= delta_t_y
    
    return traversed_pixels

            

if __name__ == "__main__":

    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[1, 1] = 1

    aa = np.array([-0.5, -0.5])
    bb = np.array([0.5, 0.5])


    for degree in range(360):
        ray_origin = np.array([np.cos(degree*np.pi/180)*RAY_SOURCE_DIST, np.sin(degree*np.pi/180)*RAY_SOURCE_DIST])

        ray_angles = np.linspace(-RAY_ANGLE/2, RAY_ANGLE/2, BEAM_NUM)
        base_direction = -ray_origin

        ms = np.tan(ray_angles)
        perpendicular_direction = np.array([-base_direction[1], base_direction[0]])
        ray_directions = base_direction + ms[:, None] * perpendicular_direction #shape (BEAM_NUM, 2)

        print(ray_directions)

        for ray_dir in ray_directions:
            t_entry, t_exit = ray_aa_bb(aa, bb, ray_origin, ray_dir)

            if t_entry == -np.inf:
                continue
            
            first_hit = ray_origin+t_entry*ray_dir
            traversed_pixels = amanatides_and_woos_traversal(first_hit, ray_dir, aa, bb)
        
        print(traversed_pixels)





