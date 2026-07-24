import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import meep as mp


class TestFarfieldPointSymmetry(unittest.TestCase):
    def _simulation(self, symmetries, geometry_center=mp.Vector3()):
        sim = mp.Simulation(
            cell_size=mp.Vector3(4, 4, 4),
            geometry_center=geometry_center,
            symmetries=symmetries,
            resolution=10,
        )
        grid_volume = sim._create_grid_volume(False)
        sim._create_symmetries(grid_volume)
        return sim

    @staticmethod
    def _farfield_stub(values):
        calls = []

        def get_farfields(near2far, points, greencyl_tol):
            calls.append(np.asarray(points).copy())
            return np.broadcast_to(values, (len(points),) + values.shape).copy()

        return get_farfields, calls

    def test_odd_mirror_reconstructs_e_and_h(self):
        sim = self._simulation(
            [mp.Mirror(mp.Y, phase=-1)], geometry_center=mp.Vector3(y=1)
        )
        values = np.arange(1, 13, dtype=np.complex128).reshape(2, 6)
        get_farfields, calls = self._farfield_stub(values)
        near2far = SimpleNamespace(nfreqs=2, swigobj=None)
        points = [mp.Vector3(0.5, 2, 0.25), mp.Vector3(0.5, 0, 0.25)]

        with mock.patch.object(
            sim, "_get_farfields_at_points_batch", side_effect=get_farfields
        ):
            fields = sim.get_farfields_at_points(near2far, points)

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0], [[0.5, 2, 0.25]])
        phase = np.array([-1, 1, -1, 1, -1, 1])
        for component_index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
            np.testing.assert_allclose(fields[name][0], values[:, component_index])
            np.testing.assert_allclose(
                fields[name][1], values[:, component_index] * phase[component_index]
            )

    def test_multiple_mirrors_share_one_representative(self):
        sim = self._simulation([mp.Mirror(mp.X, phase=1), mp.Mirror(mp.Y, phase=-1)])
        values = np.arange(1, 7, dtype=np.complex128).reshape(1, 6)
        get_farfields, calls = self._farfield_stub(values)
        near2far = SimpleNamespace(nfreqs=1, swigobj=None)
        points = [
            mp.Vector3(1, 2, 3),
            mp.Vector3(-1, 2, 3),
            mp.Vector3(1, -2, 3),
            mp.Vector3(-1, -2, 3),
        ]

        with mock.patch.object(
            sim, "_get_farfields_at_points_batch", side_effect=get_farfields
        ):
            fields = sim.get_farfields_at_points(near2far, points)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 1)
        x_phase = np.array([-1, 1, 1, 1, -1, -1])
        y_phase = np.array([-1, 1, -1, 1, -1, 1])
        expected_phases = np.stack([np.ones(6), x_phase, y_phase, x_phase * y_phase])
        actual = np.column_stack(
            [fields[name][:, 0] for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
        )
        np.testing.assert_allclose(actual, expected_phases * values[0])

    def test_symmetry_can_be_disabled(self):
        sim = self._simulation([mp.Mirror(mp.Y, phase=-1)])
        values = np.ones((1, 6), dtype=np.complex128)
        get_farfields, calls = self._farfield_stub(values)
        near2far = SimpleNamespace(nfreqs=1, swigobj=None)
        points = [mp.Vector3(y=1), mp.Vector3(y=-1)]

        with mock.patch.object(
            sim, "_get_farfields_at_points_batch", side_effect=get_farfields
        ):
            sim.get_farfields_at_points(near2far, points, use_symmetry=False)

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0], [[0, 1, 0], [0, -1, 0]])


if __name__ == "__main__":
    unittest.main()
