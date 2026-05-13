import numpy as np

from main import amanatides_and_woos_traversal, ray_aa_bb

#tangent line test:
aa = np.array([0, 0])
bb = np.array([1, 1])

r_orig = np.array([0, 2])
r_dir = np.array([1, -1])


t_entry, t_exit = ray_aa_bb(aa, bb, r_orig, r_dir)

if t_entry != -np.inf:
    print(f"intersects at: {t_entry*r_dir+r_orig}")
else:
    print("does not intersect")


#should hit in both corners:
aa = np.array([0, 0])
bb = np.array([1, 1])

r_orig = np.array([-1, -1])
r_dir = np.array([1, 1])


t_entry, t_exit = ray_aa_bb(aa, bb, r_orig, r_dir)

if t_entry != -np.inf:
    print(f"intersects at: {t_entry*r_dir+r_orig}")
else:
    print("does not intersect")


#parallel to one axis:
aa = np.array([0, 0])
bb = np.array([1, 1])

r_orig = np.array([2, 0.5])
r_dir = np.array([-1, 0])


t_entry, t_exit = ray_aa_bb(aa, bb, r_orig, r_dir)

if t_entry != -np.inf:
    print(f"intersects at: {t_entry*r_dir+r_orig}")
else:
    print("does not intersect")


#parallel to one axis with intersection exactly through whole edge:
aa = np.array([0, 0])
bb = np.array([1, 1])

r_orig = np.array([2, 0.0])
r_dir = np.array([-1, 0])

t_entry, t_exit = ray_aa_bb(aa, bb, r_orig, r_dir)

if t_entry != -np.inf:
    print(f"intersects at: {t_entry*r_dir+r_orig}")
else:
    print("does not intersect")


#ordinary intersection:
aa = np.array([-0.5, -0.5])
bb = np.array([0.5, 0.5])

r_orig = np.array([2, 0.0])
r_dir = np.array([-1, 0.25])

t_entry, t_exit = ray_aa_bb(aa, bb, r_orig, r_dir)

first_hit_coord = t_entry*r_dir+r_orig
if t_entry != -np.inf:
    print(f"intersects at: {first_hit_coord}")
else:
    print("does not intersect")


traversal_points = amanatides_and_woos_traversal(first_hit_coord, r_dir, aa, bb)

print(traversal_points)