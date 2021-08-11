from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import data_linewidth_plot


class LampParams:
    def __init__(
            self,
            sphere_diameter=450,
            layer_thickness=8,
            ring_width=15,
            sphere_opening_top=150,
            sphere_opening_bottom=250
    ):
        self.sphere_diameter = sphere_diameter
        self.layer_thickness = layer_thickness
        self.ring_width = ring_width
        self.sphere_opening_top = sphere_opening_top
        self.sphere_opening_bottom = sphere_opening_bottom


class CutArea:
    def __init__(self):
        self._rings: List[np.ndarray] = []

    @property
    def rings(self) -> List[np.ndarray]:
        return self._rings

    def get_cut_rings(self, tolerance: int = 0) -> List[np.ndarray]:
        res = [self.rings[0]]
        prev_diam = self.rings[0][2]
        for ring in self.rings[1:]:
            if ring[1] - prev_diam <= 2 * tolerance:
                ring[1] = prev_diam
            res.append(ring)

        return res

    def get_cut_diams(self, tolerance: int = 0) -> List[int]:
        res = [self.rings[0][1]]
        prev_diam = self.rings[0][2]
        for ring in self.rings[1:]:
            if ring[1] - prev_diam > 2 * tolerance:
                res.append(ring[1])
            res.append(ring[2])
            prev_diam = ring[2]

        return res

    def get_total_cut_length(self, tolerance: int = 0) -> float:
        return sum([np.pi * diam for diam in self.get_cut_diams(tolerance)])

    @property
    def diameter(self) -> int:
        return max([ring[2] for ring in self.rings])  # TODO

    def add_ring(self, ring: np.ndarray):
        self._rings.append(ring)


def compute_rings(lamps: List[LampParams]) -> np.ndarray:
    lamp_arrays = []
    for lamp_idx, lamp_params in enumerate(lamps):
        # start with center ring
        r_o = lamp_params.sphere_diameter / 2
        r_i = r_o - lamp_params.ring_width
        res = [[0, r_i, r_o, lamp_idx, 0]]

        # top rings
        z = 0
        z_idx = 0
        while r_i > lamp_params.sphere_opening_top / 2:
            z += lamp_params.layer_thickness
            z_idx += 1
            r_o = np.sqrt((lamp_params.sphere_diameter / 2) ** 2 - z ** 2)
            r_i = r_o - lamp_params.ring_width
            res = [[z, r_i, r_o, lamp_idx, z_idx]] + res

        # bottom rings
        z = 0
        z_idx = 0
        r_i = lamp_params.sphere_diameter / 2 - lamp_params.ring_width
        while r_i > lamp_params.sphere_opening_bottom / 2:
            z -= lamp_params.layer_thickness
            z_idx -= 1
            r_o = np.sqrt((lamp_params.sphere_diameter / 2) ** 2 - z ** 2)
            r_i = r_o - lamp_params.ring_width
            res += [[z, r_i, r_o, lamp_idx, z_idx]]

        # convert to numpy array and round to integers
        res = np.asarray(res)

        # correct inner diameters for sufficient overlap
        i = (len(res) - 1) // 2
        for ii in range(1, i):
            middle = (res[ii - 1, 2] + res[ii - 1, 1]) / 2
            res[ii, 1] = min(res[ii, 1], middle)
        for ii in range(i + 1, len(res) - 1):
            middle = (res[ii + 1, 2] + res[ii + 1, 1]) / 2
            res[ii, 1] = min(res[ii, 1], middle)

        res[:, 1:3] *= 2
        res = np.ceil(res).astype(np.int32)

        print(res)

        lamp_arrays.append(res)

    res = np.vstack(lamp_arrays)
    return res


def get_cut_areas(rings: np.ndarray) -> List[CutArea]:
    rings = rings.copy()

    res = []

    while len(rings):
        rings, cut_area = get_cut_area(rings)
        res.append(cut_area)

    return res


def get_cut_area(rings: np.ndarray) -> Tuple[np.ndarray, CutArea]:
    # start with smallest inner radius and corresponding outer radius for the first ring
    idx = np.argmin(rings[:, 1])

    cut_area = CutArea()
    cut_area.add_ring(rings[idx])
    rings = np.delete(rings, idx, axis=0)

    # now consecutively look for closest matching inner radius for the next ring (within a certain threshold)
    while True:
        candidates = rings[rings[:, 1] >= cut_area.diameter]
        if len(candidates) == 0:
            return rings, cut_area

        spacings = candidates[:, 1] - cut_area.diameter
        idx_c = np.argmin(spacings)
        d_i, d_o = candidates[idx_c, 1:3]
        idx = np.where(np.logical_and(rings[:, 1] == d_i, rings[:, 2] == d_o))[0][0]
        cut_area.add_ring(rings[idx])
        rings = np.delete(rings, idx, axis=0)


def main(visualize=False):
    lamps = [
        LampParams(
            sphere_diameter=260,
            layer_thickness=7,
            ring_width=15,
            sphere_opening_top=95,
            sphere_opening_bottom=115
        ),
        LampParams(
            sphere_diameter=360,
            layer_thickness=7,
            ring_width=15,
            sphere_opening_top=115,
            sphere_opening_bottom=170
        ),
        LampParams(
            sphere_diameter=450,
            layer_thickness=7,
            ring_width=15,
            sphere_opening_top=150,
            sphere_opening_bottom=235
        )
    ]
    tolerance = 4
    work_area = [940, 565]

    rings = compute_rings(lamps)
    cut_areas = get_cut_areas(rings)

    for cut_area in cut_areas:
        print(cut_area.get_cut_diams(tolerance))

    print(f"#rings (total):     {len(rings)}")
    print(f"#cuts (total):      {sum([len(cut_area.get_cut_diams(tolerance)) for cut_area in cut_areas])}")
    print(f"#cut areas:         {len(cut_areas)}")
    print(f"total cut length:   {sum([cut_area.get_total_cut_length(tolerance) for cut_area in cut_areas]) * 1e-3:.02f}m")

    rings = np.stack([ring for cut_area in cut_areas for ring in cut_area.get_cut_rings(tolerance)], axis=0)

    if visualize:
        zmin = rings[:, 0].min() / 2 - 25
        zmax = rings[:, 0].max() / 2 + 25
        xmax = rings[:, 2].max() / 2 + 25
        xmin = -xmax - 25
        fig, ax = plt.subplots(1, 3)
        plt.suptitle("cardboard lamps")
        for lamp_idx in range(len(lamps)):
            ax[lamp_idx].axis('equal')
            ax[lamp_idx].set_xlim(xmin, xmax)
            ax[lamp_idx].set_ylim(zmin, zmax)
            arr = rings[rings[:, 3] == lamp_idx][:, :3]
            for z, d_i, d_o in arr:
                for sign in {-1, 1}:
                    ax[lamp_idx].plot(
                        [sign * d_i / 2, sign * d_o / 2], [z, z],
                        linewidth=lamps[lamp_idx].layer_thickness,
                        color='brown'
                    )
        plt.show()


if __name__ == "__main__":
    main(visualize=True)
