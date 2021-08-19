import math
import os
import time
from typing import List, Optional, Tuple

import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ezdxf import units
from ezdxf.layouts import Modelspace

from utils import data_linewidth_plot


class LampParams:
    def __init__(
            self,
            sphere_diameter=450,
            layer_thickness=7,
            ring_width=15,
            sphere_opening_top=150,
            sphere_opening_bottom=250,
            socket_diam=40,
            socket_ring_width=20,
            strut_width=60,
            socket_layer_indices=None,
            bulb_length=170,
            bulb_diameter=125
    ):
        self.sphere_diameter = sphere_diameter
        self.layer_thickness = layer_thickness
        self.ring_width = ring_width
        self.sphere_opening_top = sphere_opening_top
        self.sphere_opening_bottom = sphere_opening_bottom
        self.socket_diam = socket_diam
        self.socket_ring_width = socket_ring_width
        self.strut_width = strut_width
        self.socket_layer_indices = socket_layer_indices if socket_layer_indices is not None else [3, 4]
        self.bulb_length = bulb_length
        self.bulb_diameter = bulb_diameter

    @property
    def socket_length(self) -> int:
        return self.bulb_length - self.bulb_diameter


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
    def inner_diameter(self) -> int:
        return min([ring[1] for ring in self.rings])

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


class SocketCutarea(CutArea):
    def __init__(
            self,
            *args,
            socket_diam: float,
            socket_ring_width: float,
            strut_width: float,
            outer_ring: np.ndarray,
            outer_ring_corner_diam: float = 0,
            socket_ring_corner_diam: float = 0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.socket_diam = socket_diam
        self.socket_ring_width = socket_ring_width
        self.strut_width = strut_width
        self.outer_ring_corner_diam = outer_ring_corner_diam
        self.socket_ring_corner_diam = socket_ring_corner_diam
        self._rings = [outer_ring]

    @property
    def inner_diameter(self) -> int:
        return 0

    @property
    def cut_diams(self) -> List[int]:
        res = [self.rings[0][2]]
        prev_diam = self.rings[0][2]
        for ring in self.rings[1:]:
            if ring[1] - prev_diam > 2 * self.tolerance:
                res.append(ring[1])
            res.append(ring[2])
            prev_diam = ring[2]

        return res

    @property
    def total_cut_length(self) -> float:
        res = sum([np.pi * diam for diam in self.cut_diams])
        return res  # TODO add arc lenghs of inner cuts

    def _get_points_outer(self):
        r_outer = self.rings[0][1] / 2
        y_strut = self.strut_width / 2
        r_corner = self.outer_ring_corner_diam / 2
        y_corner = y_strut + r_corner
        p_corner = np.asarray([np.sqrt((r_outer - r_corner) ** 2 - y_corner ** 2), y_corner])
        p_strut = np.asarray([p_corner[0], y_strut])
        p_outer = p_corner / np.linalg.norm(p_corner) * r_outer

        return p_strut, p_outer, p_corner

    def _get_points_inner(self):
        r_inner = self.socket_diam / 2 + self.socket_ring_width
        y_strut = self.strut_width / 2
        r_corner = self.socket_ring_corner_diam / 2
        y_corner = y_strut + r_corner
        p_corner = np.asarray([np.sqrt((r_inner + r_corner) ** 2 - y_corner ** 2), y_corner])
        p_strut = np.asarray([p_corner[0], y_strut])
        p_inner = p_corner / np.linalg.norm(p_corner) * r_inner

        return p_strut, p_inner, p_corner

    def plot(self, ax=None, x0=None):
        if x0 is None:
            x0 = np.zeros([2])
        center = np.asarray([self.diameter / 2] * 2) + x0
        if ax is None:
            fig, ax = plt.subplots()

        for diam in self.cut_diams:
            circle = plt.Circle(center, diam / 2, color='black', fill=False)
            ax.add_patch(circle)

        # plot inner socket ring
        circle = plt.Circle(center, self.socket_diam / 2, color='black', fill=False)
        ax.add_patch(circle)

        # compute intersection points
        p_strut_outer, p_outer, p_corner_outer = self._get_points_outer()
        p_strut_inner, p_inner, p_corner_inner = self._get_points_inner()

        # plot rings (socket and outer)
        d_inner = self.socket_diam + 2 * self.socket_ring_width
        d_outer = self.rings[0][1]
        for d, p in zip([d_inner, d_outer], [p_inner, p_outer]):
            for sign in {-1, 1}:
                p1 = [p[0], sign * p[1]]
                p2 = [-p[0], sign * p[1]]
                theta1 = math.atan2(p1[1], p1[0]) * 180 / np.pi
                theta2 = math.atan2(p2[1], p2[0]) * 180 / np.pi
                if sign == -1:
                    theta1, theta2 = theta2, theta1
                arc = patches.Arc(center, d, d, angle=0.0, theta1=theta1, theta2=theta2)
                ax.add_patch(arc)

        # plot struts
        for x_sign in {-1, 1}:
            for y_sign in {-1, 1}:
                line = plt.Line2D(
                    [center[0] + x_sign * p_strut_inner[0], center[0] + x_sign * p_strut_outer[0]],
                    [center[1] + y_sign * p_strut_inner[1], center[1] + y_sign * p_strut_inner[1]],
                    color='black'
                )
                ax.add_line(line)

        # plot corner circles (if needed)
        for p_strut, p_circle, p_corner, diam, theta_sign in zip(
                [p_strut_outer, p_strut_inner],
                [p_outer, p_inner],
                [p_corner_outer, p_corner_inner],
                [self.outer_ring_corner_diam, self.socket_ring_corner_diam],
                [1, -1]
        ):
            if diam <= 0:
                continue

            for x_sign in {-1, 1}:
                for y_sign in {-1, 1}:
                    v_sign = np.asarray([x_sign, y_sign])
                    p1 = (p_strut - p_corner) * v_sign
                    p2 = (p_circle - p_corner) * v_sign
                    theta1 = math.atan2(p1[1], p1[0]) * 180 / np.pi
                    theta2 = math.atan2(p2[1], p2[0]) * 180 / np.pi
                    if x_sign * y_sign * theta_sign == -1:
                        theta1, theta2 = theta2, theta1
                    arc = patches.Arc(center + p_corner * v_sign, diam, diam, angle=0.0, theta1=theta1, theta2=theta2)
                    ax.add_patch(arc)

    def add_to_dxf(self, msp: Modelspace, x0=None, layer: str = 'background'):
        if x0 is None:
            x0 = np.zeros([2])
        center = np.asarray([self.diameter / 2] * 2) + x0

        for diam in self.cut_diams:
            msp.add_circle(center, diam / 2)

        # TODO: add inner stuff to dxf


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


def split_rings(rings: np.ndarray, lamps: List[LampParams]) -> Tuple[np.ndarray, np.ndarray]:
    strut_rings = []
    new_rings = None
    for lamp_idx, lamp in enumerate(lamps):
        lamp_rings = rings[rings[:, 3] == lamp_idx]
        lamp_rings = lamp_rings[lamp_rings[:, -1].argsort()]
        lamp_rings = np.flip(lamp_rings, axis=0)
        for idx in lamp.socket_layer_indices:
            strut_rings.append(lamp_rings[idx])
        lamp_rings = np.delete(lamp_rings, lamp.socket_layer_indices, axis=0)
        if new_rings is None:
            new_rings = lamp_rings
        else:
            new_rings = np.vstack([new_rings, lamp_rings])

    return new_rings, np.vstack(strut_rings)


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


def merge_cut_areas(cut_areas: List[CutArea]) -> List[CutArea]:
    max_inner_diam = max([cut_area.inner_diameter for cut_area in cut_areas])
    fitting_outer_diams = [cut_area.diameter for cut_area in cut_areas if cut_area.diameter <= max_inner_diam]
    while fitting_outer_diams:
        outer_diam = max(fitting_outer_diams)
        cut_area = [cut_area for cut_area in cut_areas if cut_area.diameter == outer_diam][0]
        other = [cut_area for cut_area in cut_areas if cut_area.inner_diameter == max_inner_diam][0]
        cut_area._rings += other.rings
        cut_areas.remove(other)

        max_inner_diam = max([cut_area.inner_diameter for cut_area in cut_areas])
        fitting_outer_diams = [cut_area.diameter for cut_area in cut_areas if cut_area.diameter <= max_inner_diam]

    return cut_areas


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
        ax.set_aspect('equal', 'box')
        if self.work_area[0] == np.infty:
            size = self.size
        else:
            size = self.work_area

        ax.set_xlim((0, size[0]))
        ax.set_ylim((0, size[1]))

        for cut_area, corner in zip(self._cut_areas, self._top_left_corners):
            cut_area.plot(ax=ax, x0=corner)

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
            sphere_opening_bottom=115,
            socket_diam=40,
            socket_ring_width=20,
            strut_width=60,
            socket_layer_indices=[5, 6],
            bulb_diameter=95,
            bulb_length=135
        ),
        LampParams(
            sphere_diameter=360,
            layer_thickness=7,
            ring_width=15,
            sphere_opening_top=115,
            sphere_opening_bottom=170,
            socket_diam=40,
            socket_ring_width=20,
            strut_width=60,
            socket_layer_indices=[10, 11],
            bulb_diameter=95,
            bulb_length=135
        ),
        LampParams(
            sphere_diameter=450,
            layer_thickness=7,
            ring_width=15,
            sphere_opening_top=150,
            sphere_opening_bottom=235,
            socket_diam=40,
            socket_ring_width=20,
            strut_width=60,
            socket_layer_indices=[13, 14],
            bulb_diameter=125,
            bulb_length=175
        )
    ]
    tolerance = 4
    padding = 5
    work_area = np.asarray([2000, 1000])

    rings = compute_rings(lamps)

    rings, strut_rings = split_rings(rings, lamps)
    cut_areas = get_cut_areas(rings, tolerance=tolerance, padding=padding)
    for lamp_idx, lamp in enumerate(lamps):
        for ring in strut_rings[strut_rings[:, 3] == lamp_idx]:
            cut_areas.append(
                SocketCutarea(
                    socket_diam=lamp.socket_diam,
                    socket_ring_width=lamp.socket_ring_width,
                    strut_width=lamp.strut_width,
                    outer_ring=ring
                )
            )
    cut_areas = merge_cut_areas(cut_areas)

    all_rings = np.stack([ring for cut_area in cut_areas for ring in cut_area.cut_rings], axis=0)

    for cut_area in cut_areas:
        print(cut_area.cut_diams)

    for lamp_idx, lamp in enumerate(lamps):
        print(f"Lamp #{lamp_idx + 1} has {len(all_rings[all_rings[:, 3] == lamp_idx])} rings")

    print(f"#rings (total):     {len(all_rings)}")
    print(f"#cuts (total):      {sum([len(cut_area.cut_diams) for cut_area in cut_areas])}")
    print(f"#cut areas:         {len(cut_areas)}")
    print(f"total cut length:   {sum([cut_area.total_cut_length for cut_area in cut_areas]) * 1e-3:.02f}m")

    if visualize:
        zmin = all_rings[:, 0].min() - 25
        zmax = all_rings[:, 0].max() + 25
        xmax = all_rings[:, 2].max() / 2 + 25
        xmin = -xmax
        fig, ax = plt.subplots(1, 3)
        plt.suptitle("cardboard lamps")
        for lamp_idx, lamp in enumerate(lamps):
            ax[lamp_idx].set_aspect('equal', 'box')
            ax[lamp_idx].set_xlim(xmin, xmax)
            ax[lamp_idx].set_ylim(zmin, zmax)
            arr = rings[rings[:, 3] == lamp_idx]
            for z, d_i, d_o, *_ in arr:
                for sign in {-1, 1}:
                    ax[lamp_idx].plot(
                        [sign * d_i / 2, sign * d_o / 2], [z, z],
                        linewidth=lamp.layer_thickness,
                        color='brown'
                    )
            for z, _, d_o, *_ in strut_rings[strut_rings[:, 3] == lamp_idx]:
                ax[lamp_idx].plot(
                    [-d_o / 2, d_o / 2], [z, z],
                    linewidth=lamp.layer_thickness,
                    color='brown'
                )

            z_strut = min(strut_rings[strut_rings[:, 3] == lamp_idx][:, 0]) - lamp.layer_thickness / 2
            for sign in {-1, 1}:
                ax[lamp_idx].plot(
                    [sign * 27 / 2, sign * 27 / 2], [z_strut, z_strut - lamp.socket_length],
                    linewidth=1,
                    color='black'
                )
            center = [0, z_strut - lamp.socket_length - lamp.bulb_diameter / 2]
            circle = plt.Circle(center, lamp.bulb_diameter / 2, color='yellow', fill=True)
            ax[lamp_idx].add_patch(circle)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        generate_drawing(cut_areas, work_area, out_dir=os.path.join("output", timestr))

        plt.show()


def show_socket_cut_area():
    cut_area = SocketCutarea(
        socket_diam=40,
        socket_ring_width=25,
        strut_width=40,
        outer_ring=np.asarray([0, 200, 215, 0, 0, 0]),
        outer_ring_corner_diam=25,
        socket_ring_corner_diam=10,
        tolerance=0,
        padding=0
    )

    fig, ax = plt.subplots(1)
    fig.suptitle("technical drawing")

    ax.set_aspect('equal', 'box')
    ax.set_xlim((0, cut_area.diameter))
    ax.set_ylim((0, cut_area.diameter))

    cut_area.plot(ax=ax, x0=np.zeros(2))

    plt.show()


if __name__ == "__main__":
    # main(visualize=True)
    show_socket_cut_area()