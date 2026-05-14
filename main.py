from dataclasses import dataclass
from typing import List

import numpy as np

RAY_SOURCE_DIST = 2
RAY_ANGLE = np.pi/4
BEAM_NUM = 5
GRID_SIZE = 2


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

def amanatides_and_woos_traversal(first_hit_coord, ray_dir, aa, bb) -> List[TraversedPixel]:
    step = np.sign(ray_dir)
    step_x = int(step[0])
    step_y = int(step[1])

    pixel_length = 1/GRID_SIZE #length of square is 1

    delta_t = np.abs(pixel_length/ray_dir)
    delta_t_x = delta_t[0]
    delta_t_y = delta_t[1]

    if step_x > 0:
        x_upper_bound = np.floor((first_hit_coord[0] + step_x*pixel_length-aa[0])/pixel_length)*pixel_length+aa[0]
        t_x = (x_upper_bound - first_hit_coord[0])/ray_dir[0]

        cur_grid_j = int(np.floor((first_hit_coord[0] - aa[0])/pixel_length))
    elif step_x <0:
        x_lower_bound = np.ceil((first_hit_coord[0] + step_x*pixel_length-aa[0])/pixel_length)*pixel_length+aa[0]
        t_x = (x_lower_bound - first_hit_coord[0])/ray_dir[0]

        cur_grid_j = int(np.ceil((first_hit_coord[0] - aa[0])/pixel_length-1))
    else:
        t_x = np.inf
        cur_grid_j = int(np.floor((first_hit_coord[0] - aa[0])/pixel_length)) #just decide for one cell as the ray goes exactly between to cells in parallel


    if step_y > 0:
        y_upper_bound = np.floor((first_hit_coord[1] + step_y*pixel_length-aa[1])/pixel_length)*pixel_length+aa[1]
        t_y = (y_upper_bound - first_hit_coord[1])/ray_dir[1]

        cur_grid_i = int(GRID_SIZE-1 - np.floor((first_hit_coord[1] - aa[1])/pixel_length))
    elif step_y < 0:
        y_lower_bound = np.ceil((first_hit_coord[1] + step_y*pixel_length-aa[1])/pixel_length)*pixel_length+aa[1]
        t_y = (y_lower_bound - first_hit_coord[1])/ray_dir[1]

        cur_grid_i = int(GRID_SIZE - np.ceil((first_hit_coord[1] - aa[1])/pixel_length))
    else:
        t_y = np.inf
        cur_grid_i = int(GRID_SIZE-1 - np.floor((first_hit_coord[1] - aa[1])/pixel_length))
    
    traversed_pixels = []
    
    prev_t = 0

    while True:
        if t_x < t_y:

            len_in_cell = np.linalg.norm((t_x - prev_t)*ray_dir)

            traversed_pixels.append(TraversedPixel(cur_grid_i, cur_grid_j, len_in_cell))

            cur_grid_j += step_x

            if cur_grid_j < 0 or cur_grid_j >= GRID_SIZE:
                break

            prev_t = t_x
            t_x += delta_t_x
            
        elif t_x == t_y: #maybe also using np.is_close with tolerance

            len_in_cell = np.linalg.norm((t_x - prev_t)*ray_dir)

            traversed_pixels.append(TraversedPixel(cur_grid_i, cur_grid_j, len_in_cell))

            cur_grid_j += step_x
            cur_grid_i -= step_y

            if cur_grid_i < 0 or cur_grid_i >= GRID_SIZE or cur_grid_j < 0 or cur_grid_j >= GRID_SIZE:
                break
        else:
            len_in_cell = np.linalg.norm((t_y - prev_t)*ray_dir)
            traversed_pixels.append(TraversedPixel(cur_grid_i, cur_grid_j, len_in_cell))

            cur_grid_i -= step_y

            if cur_grid_i < 0 or cur_grid_i >= GRID_SIZE:
                break

            prev_t = t_y
            t_y+= delta_t_y
    
    return traversed_pixels


def flatten_i_j(i, j):
    ret = i*GRID_SIZE + j
    return ret


if __name__ == "__main__":

    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[1, 1] = 1

    aa = np.array([-0.5, -0.5])
    bb = np.array([0.5, 0.5])

    lin_eq_holder = np.zeros((360*BEAM_NUM, GRID_SIZE**2))
    b = np.zeros(360*BEAM_NUM)

    i = 0

    for degree in range(360):
        ray_origin = np.array([np.cos(degree*np.pi/180)*RAY_SOURCE_DIST, np.sin(degree*np.pi/180)*RAY_SOURCE_DIST])

        ray_angles = np.linspace(-RAY_ANGLE/2, RAY_ANGLE/2, BEAM_NUM)
        base_direction = -ray_origin

        ms = np.tan(ray_angles)
        perpendicular_direction = np.array([-base_direction[1], base_direction[0]])
        ray_directions = base_direction + ms[:, None] * perpendicular_direction #shape (BEAM_NUM, 2)

        #print(ray_directions)

        for ray_dir in ray_directions:
            t_entry, t_exit = ray_aa_bb(aa, bb, ray_origin, ray_dir)

            if t_entry == -np.inf:
                continue
            
            first_hit = np.clip(ray_origin+t_entry*ray_dir, -0.5, 0.5) #the clip ensures that floating point inaccuracies dont result in components like -0.5000...1 which would then result in out of bounds grid coords like i = GRID_SIZE
            #maybe put clip in function for better testing

            traversed_pixels = amanatides_and_woos_traversal(first_hit, ray_dir, aa, bb)

            out_ray_val = 0
            for tp in traversed_pixels:
                out_ray_val += grid[tp.i, tp.j]*tp.length
                lin_eq_holder[i, flatten_i_j(tp.i, tp.j)] = tp.length
                #todo grow the lin_eq_holder and b dynamically to not have zero lines if the ray doesnt hit the object

            b[i] = out_ray_val
            i += 1
            print(i)
    
    print(lin_eq_holder)
    print(b)





