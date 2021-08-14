import os
import time
from typing import List, Optional, Tuple

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from ezdxf import units
from ezdxf.layouts import Modelspace

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
    def __init__(self, tolerance: float = 0, padding: float = 0):
        self.tolerance = tolerance
        self.padding = padding
        self._rings: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._rings)

    @property
    def rings(self) -> List[np.ndarray]:
        return self._rings

    @property
    def cut_rings(self) -> List[np.ndarray]:
        res = [self.rings[0]]
        prev_diam = self.rings[0][2]
        for ring in self.rings[1:]:
            if ring[1] - prev_diam <= 2 * self.tolerance:
                ring[1] = prev_diam
            res.append(ring)

        return res

    @property
    def cut_diams(self) -> List[int]:
        res = [self.rings[0][1], self.rings[0][2]]
        prev_diam = self.rings[0][2]
        for ring in self.rings[1:]:
            if ring[1] - prev_diam > 2 * self.tolerance:
                res.append(ring[1])
            res.append(ring[2])
            prev_diam = ring[2]

        return res

    @property
    def total_cut_length(self) -> float:
        return sum([np.pi * diam for diam in self.cut_diams])

    @property
    def diameter(self) -> int:
        return max([ring[2] for ring in self.rings])

    @property
    def job_diameter(self) -> int:
        return max([ring[2] for ring in self.rings]) + 2 * self.padding

    def add_ring(self, ring: np.ndarray):
        self._rings.append(ring)

    def plot(self, ax=None, x0=None):
        if x0 is None:
            x0 = np.zeros([2])
        center = np.asarray([self.diameter / 2] * 2) + x0
        if ax is None:
            fig, ax = plt.subplots()

        for diam in self.cut_diams:
            circle = plt.Circle(center, diam / 2, color='black', fill=False)
            ax.add_patch(circle)

    def add_to_dxf(self, msp: Modelspace, x0=None, layer: str = 'background'):
        if x0 is None:
            x0 = np.zeros([2])
        center = np.asarray([self.diameter / 2] * 2) + x0

        for diam in self.cut_diams:
            msp.add_circle(center, diam / 2)


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


def get_cut_areas(rings: np.ndarray, **kwargs) -> List[CutArea]:
    rings = rings.copy()

    res = []

    while len(rings):
        rings, cut_area = get_cut_area(rings, **kwargs)
        res.append(cut_area)

    return res


def get_cut_area(rings: np.ndarray, **kwargs) -> Tuple[np.ndarray, CutArea]:
    # start with smallest inner radius and corresponding outer radius for the first ring
    idx = np.argmin(rings[:, 1])

    cut_area = CutArea(**kwargs)
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


class Job:
    def __init__(self, work_area: Optional[np.ndarray] = None):
        if work_area is None:
            work_area = np.asarray([np.infty] * 2)

        self.work_area = np.asarray([work_area[:].max(), work_area[:].min()])
        self._cut_areas = []
        self._top_left_corners = []

    def __iter__(self):
        return self._cut_areas.__iter__()

    @property
    def size(self) -> np.ndarray:
        return self._top_left_corners[-1] + self._cut_areas[-1].diameter

    def assign_cut_areas(self, cut_areas: List[CutArea]) -> List[CutArea]:
        if max([cut_area.diameter for cut_area in cut_areas]) > self.work_area[:].min():
            raise ValueError("Work area is too small!")

        cas = cut_areas[:]

        x0, y0 = 0, 0
        row_max_diam = 0
        while True:
            # first, check if we can still fit any cut area into this row or the next row
            candidates = [
                ca for ca in cas
                if (
                        (x0 + ca.job_diameter < self.work_area[0] and y0 + ca.job_diameter < self.work_area[1]) or
                        (y0 + row_max_diam + ca.job_diameter < self.work_area[1])
                )
            ]
            if not candidates:
                break

            # now, check if we can still fit a cut area in this row
            candidates = [
                ca for ca in cas
                if x0 + ca.job_diameter < self.work_area[0] and y0 + ca.job_diameter < self.work_area[1]
            ]
            if not candidates:
                # go to the next row
                y0 += row_max_diam
                x0 = 0
                row_max_diam = 0
                continue

            # of all candidates, first select those with maximum diameter
            max_diam = max([ca.job_diameter for ca in candidates])
            candidates = [ca for ca in candidates if ca.job_diameter == max_diam]

            # now, select the cut area with the maximum number of cuts
            ca = max(candidates, key=lambda ca: len(ca))
            cas.remove(ca)
            self._cut_areas.append(ca)
            self._top_left_corners.append(np.asarray([x0, y0]))
            row_max_diam = max(row_max_diam, ca.job_diameter)
            x0 += ca.job_diameter

        return cas

    def plot(self, ax):
        ax.axis('equal')

        if self.work_area[0] == np.infty:
            x0 = np.zeros(2)
            size = self.size
        else:
            x0 = (self.work_area - self.size) / 2
            size = self.work_area

        ax.set_xlim((0, size[0]))
        ax.set_ylim((0, size[1]))

        for cut_area, corner in zip(self._cut_areas, self._top_left_corners):
            cut_area.plot(ax=ax, x0=x0 + corner)

    def save_dxf(self, filename: str):
        doc = ezdxf.new()
        # Set centimeter as document/modelspace units
        doc.units = units.MM
        # which is a shortcut (including validation) for
        doc.header['$INSUNITS'] = units.MM
        msp = doc.modelspace()

        for cut_area, corner in zip(self._cut_areas, self._top_left_corners):
            cut_area.add_to_dxf(msp=msp)

        doc.saveas(filename)


def get_jobs(cut_areas: List[CutArea], work_area: Optional[np.ndarray] = None) -> List[Job]:
    if max([cut_area.diameter for cut_area in cut_areas]) > work_area[:].min():
        raise ValueError("Work area is too small!")

    cas = cut_areas[:]
    jobs = []
    while cas:
        job = Job(work_area)
        cas = job.assign_cut_areas(cas)
        jobs.append(job)

    return jobs


def generate_drawing(cut_areas: List[CutArea], work_area=None, out_dir: str = None):
    if work_area is None:
        work_area = np.asarray([np.infty] * 2)

    jobs = get_jobs(cut_areas, work_area)

    fig, axs = plt.subplots(len(jobs))
    fig.suptitle("technical drawing")
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])
    for idx, (job, ax) in enumerate(zip(jobs, axs)):
        job.plot(ax)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            job.save_dxf(os.path.join(out_dir, f"job_{idx + 1}.dxf"))


def main(visualize=False):
    lamps = [
        LampParams(
            sphere_diameter=270,
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
    padding = 5
    work_area = np.asarray([4000, 2000])

    rings = compute_rings(lamps)
    cut_areas = get_cut_areas(rings, tolerance=tolerance, padding=padding)

    for cut_area in cut_areas:
        print(cut_area.cut_diams)

    print(f"#rings (total):     {len(rings)}")
    print(f"#cuts (total):      {sum([len(cut_area.cut_diams) for cut_area in cut_areas])}")
    print(f"#cut areas:         {len(cut_areas)}")
    print(f"total cut length:   {sum([cut_area.total_cut_length for cut_area in cut_areas]) * 1e-3:.02f}m")

    rings = np.stack([ring for cut_area in cut_areas for ring in cut_area.cut_rings], axis=0)

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

        timestr = time.strftime("%Y%m%d-%H%M%S")
        generate_drawing(cut_areas, work_area, out_dir=os.path.join("output", timestr))

        plt.show()


if __name__ == "__main__":
    main(visualize=True)
