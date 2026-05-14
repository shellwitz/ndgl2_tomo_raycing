import math
import unittest

import numpy as np

from main import amanatides_and_woos_traversal, ray_aa_bb


AA = np.array([-0.5, -0.5])
BB = np.array([0.5, 0.5])


def first_hit(origin, direction, aa=AA, bb=BB):
    t_entry, t_exit = ray_aa_bb(aa, bb, origin, direction)
    if t_entry == -np.inf:
        raise AssertionError(f"ray missed box: origin={origin}, direction={direction}")
    return origin + t_entry * direction


def traversal_from_ray(origin, direction, aa=AA, bb=BB):
    hit = first_hit(origin, direction, aa, bb)
    return amanatides_and_woos_traversal(hit, direction, aa, bb)


def simplify(traversed_pixels):
    return [
        (pixel.i, pixel.j, float(pixel.length))
        for pixel in traversed_pixels
    ]


def assert_traversal(test_case, actual, expected_cells, expected_lengths):
    actual = simplify(actual)
    actual_cells = [(i, j) for i, j, _ in actual]
    actual_lengths = [length for _, _, length in actual]

    test_case.assertEqual(
        actual_cells,
        expected_cells,
        f"wrong cells; full traversal was {actual}",
    )
    test_case.assertEqual(
        len(actual_lengths),
        len(expected_lengths),
        f"wrong number of lengths; full traversal was {actual}",
    )

    for actual_length, expected_length in zip(actual_lengths, expected_lengths):
        test_case.assertAlmostEqual(actual_length, expected_length)


class RayBoxIntersectionTests(unittest.TestCase):
    def test_tangent_line_does_not_count_as_area_intersection(self):
        aa = np.array([0.0, 0.0])
        bb = np.array([1.0, 1.0])
        origin = np.array([0.0, 2.0])
        direction = np.array([1.0, -1.0])

        self.assertEqual(ray_aa_bb(aa, bb, origin, direction), (-np.inf, -np.inf))

    def test_corner_to_corner_intersection_counts(self):
        aa = np.array([0.0, 0.0])
        bb = np.array([1.0, 1.0])
        origin = np.array([-1.0, -1.0])
        direction = np.array([1.0, 1.0])

        t_entry, t_exit = ray_aa_bb(aa, bb, origin, direction)

        np.testing.assert_allclose(origin + t_entry * direction, np.array([0.0, 0.0]))
        np.testing.assert_allclose(origin + t_exit * direction, np.array([1.0, 1.0]))

    def test_axis_parallel_ray_intersects_box(self):
        aa = np.array([0.0, 0.0])
        bb = np.array([1.0, 1.0])
        origin = np.array([2.0, 0.5])
        direction = np.array([-1.0, 0.0])

        t_entry, t_exit = ray_aa_bb(aa, bb, origin, direction)

        np.testing.assert_allclose(origin + t_entry * direction, np.array([1.0, 0.5]))
        np.testing.assert_allclose(origin + t_exit * direction, np.array([0.0, 0.5]))


class AmanatidesAndWooTraversalTests(unittest.TestCase):
    def test_ordinary_ray_crosses_two_cells(self):
        origin = np.array([2.0, 0.0])
        direction = np.array([-1.0, 0.3])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(0, 1)],
            expected_lengths=[math.sqrt(1.09) / 6],
        )

    def test_horizontal_ray_crosses_top_row_from_right_to_left(self):
        origin = np.array([1.0, 0.25])
        direction = np.array([-1.0, 0.0])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(0, 1), (0, 0)],
            expected_lengths=[0.5, 0.5],
        )

    def test_vertical_ray_crosses_right_column_from_top_to_bottom(self):
        origin = np.array([0.25, 1.0])
        direction = np.array([0.0, -1.0])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(0, 1), (1, 1)],
            expected_lengths=[0.5, 0.5],
        )

    def test_corner_crossing_steps_x_and_y_together(self):
        origin = np.array([-1.0, -1.0])
        direction = np.array([1.0, 1.0])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(1, 0), (0, 1)],
            expected_lengths=[math.sqrt(0.5), math.sqrt(0.5)],
        )

    def test_internal_horizontal_boundary_uses_cell_above_when_ray_moves_up(self):
        origin = np.array([-1.0, -0.1])
        direction = np.array([1.0, 0.2])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(0, 0), (0, 1)],
            expected_lengths=[math.sqrt(0.26), math.sqrt(0.26)],
        )

    def test_internal_horizontal_boundary_uses_cell_below_when_ray_moves_down(self):
        origin = np.array([-1.0, 0.1])
        direction = np.array([1.0, -0.2])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(1, 0), (1, 1)],
            expected_lengths=[math.sqrt(0.26), math.sqrt(0.26)],
        )

    def test_internal_vertical_boundary_uses_cell_left_when_ray_moves_left(self):
        origin = np.array([0.1, -1.0])
        direction = np.array([-0.2, 1.0])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(1, 0), (0, 0)],
            expected_lengths=[math.sqrt(0.26), math.sqrt(0.26)],
        )

    def test_internal_vertical_boundary_uses_cell_right_when_ray_moves_right(self):
        origin = np.array([-0.1, -1.0])
        direction = np.array([0.2, 1.0])

        actual = traversal_from_ray(origin, direction)

        assert_traversal(
            self,
            actual,
            expected_cells=[(1, 1), (0, 1)],
            expected_lengths=[math.sqrt(0.26), math.sqrt(0.26)],
        )

#todo take the actual not cutoff values for i=682 in main -> the amanatides_and_woos_traversal works if the first_hit_coord is np.clip before
    def test_near_corner_entry_stays_in_grid(self):
        first_hit_coord = np.array([-0.46625754, -0.5])
        direction = np.array([1.36399672, 1.4627074])

        actual = amanatides_and_woos_traversal(first_hit_coord, direction, AA, BB)

        assert_traversal(
            self,
            actual,
            expected_cells=[(1, 0), (1, 1), (0, 1)],
            expected_lengths=[
                0.6836637252734603,
                5.9506995924434355e-09,
                0.6836637312241599,
            ],
        )

if __name__ == "__main__":
    unittest.main()
