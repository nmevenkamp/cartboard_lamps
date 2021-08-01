from typing import List

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


def compute_rings(lamps: List[LampParams]) -> np.ndarray:
    lamp_arrays = []
    for lamp_idx, lamp_params in enumerate(lamps):
        # start with center ring
        r_o = lamp_params.sphere_diameter / 2
        r_i = r_o - lamp_params.ring_width
        res = [[0, r_i, r_o, lamp_idx]]

        # top rings
        z = 0
        while r_i > lamp_params.sphere_opening_top / 2:
            z += lamp_params.layer_thickness
            r_o = np.sqrt((lamp_params.sphere_diameter / 2) ** 2 - z ** 2)
            r_i = r_o - lamp_params.ring_width
            res = [[z, r_i, r_o, lamp_idx]] + res

        # bottom rings
        z = 0
        r_i = lamp_params.sphere_diameter / 2 - lamp_params.ring_width
        while r_i > lamp_params.sphere_opening_bottom / 2:
            z -= lamp_params.layer_thickness
            r_o = np.sqrt((lamp_params.sphere_diameter / 2) ** 2 - z ** 2)
            r_i = r_o - lamp_params.ring_width
            res += [[z, r_i, r_o, lamp_idx]]

        # convert to numpy array and round to integers
        res = np.asarray(res)
        res = np.round(res)

        # correct inner diameters for sufficient overlap
        i = (len(res) - 1) // 2
        for ii in range(1, i):
            middle = int(np.ceil((res[ii - 1, 2] + res[ii - 1, 1]) / 2))
            res[ii, 1] = min(res[ii, 1], middle)
        for ii in range(i + 1, len(res) - 1):
            middle = int(np.ceil((res[ii + 1, 2] + res[ii + 1, 1]) / 2))
            res[ii, 1] = min(res[ii, 1], middle)

        lamp_arrays.append(res)

    res = np.vstack(lamp_arrays)
    return res


def post_process_rings(rings: np.ndarray) -> np.ndarray:
    r_os = np.unique(rings[:, 2])
    for r_o in r_os:
        idxs = rings[:, 2] == r_o
        r_i = min(rings[idxs, 1])
        rings[idxs, 1] = r_i

    return rings


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

    rings = compute_rings(lamps)
    rings = post_process_rings(rings)

    rings_sorted_ro = rings[rings[:, 2].argsort()]
    print(rings_sorted_ro)
    print(f"#rings: {len(rings)}")

    if visualize:
        zmin = rings[:, 0].min()
        zmax = rings[:, 0].max()
        xmax = rings[:, 2].max()
        xmin = -xmax
        fig, ax = plt.subplots(1, 3)
        plt.suptitle("cardboard lamps")
        for lamp_idx in range(len(lamps)):
            ax[lamp_idx].axis('equal')
            arr = rings[rings[:, 3] == lamp_idx][:, :3]
            for z, r_i, r_o in arr:
                data_linewidth_plot([r_i, r_o], [z, z], ax=ax[lamp_idx], linewidth=lamps[lamp_idx].layer_thickness, color='brown')
                data_linewidth_plot([-r_i, -r_o], [z, z], ax=ax[lamp_idx], linewidth=lamps[lamp_idx].layer_thickness, color='brown')
        plt.show()


if __name__ == "__main__":
    main(visualize=True)
