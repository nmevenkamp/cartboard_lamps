import math
import os
import time
from io import TextIOWrapper
from typing import List, Optional, Tuple
from zipfile import ZipFile

import ezdxf
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ezdxf import units
from ezdxf.layouts import Modelspace


class LampParams:
    def __init__(
            self,
            sphere_diameter: float = 450,
            layer_thickness: float = 7,
            ring_width: float = 15,
            sphere_opening_top: float = 150,
            sphere_opening_bottom: float = 250,
            socket_diam: float = 40,
            socket_ring_width: float = 20,
            strut_width: float = 60,
            outer_ring_corner_diam: float = 25,
            socket_ring_corner_diam: float = 10,
            socket_layer_indices: Optional[List[int]] = None,
            bulb_length: float = 170,
            bulb_diameter: float = 125
    ):
        self.sphere_diameter = sphere_diameter
        self.layer_thickness = layer_thickness
        self.ring_width = ring_width
        self.sphere_opening_top = sphere_opening_top
        self.sphere_opening_bottom = sphere_opening_bottom
        self.socket_diam = socket_diam
        self.socket_ring_width = socket_ring_width
        self.strut_width = strut_width
        self.outer_ring_corner_diam = outer_ring_corner_diam
        self.socket_ring_corner_diam = socket_ring_corner_diam
        self.socket_layer_indices = socket_layer_indices if socket_layer_indices is not None else [3, 4]
        self.bulb_length = bulb_length
        self.bulb_diameter = bulb_diameter

    @property
    def socket_length(self) -> float:
        return self.bulb_length - self.bulb_diameter


class Ring:
    def __init__(
            self,
            diam_inner: float,
            diam_outer: float,
            has_socket: bool = False,
            lamp_idx: int = None,
            z: float = None,
            z_idx: int = None
    ):
        self.diam_inner = diam_inner
        self.diam_outer = diam_outer
        self.has_socket = has_socket
        self.lamp_idx = lamp_idx
        self.z = z
        self.z_idx = z_idx


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

    def add_rings(self, rings: List[np.ndarray]):
        self._rings += rings

    def plot(self, ax=None, x0=None):
        if ax is None:
            _, ax = plt.subplots()

        self._draw(ax, x0)

    def add_to_dxf(self, msp: Modelspace, x0=None, layer: str = 'background') -> List[Ring]:
        return self._draw(msp, x0)

    def _draw(self, obj, x0) -> List[Ring]:
        if x0 is None:
            x0 = np.zeros([2])
        center = np.asarray([self.diameter / 2] * 2) + x0

        rings = []
        d_o = None

        for diam in self.cut_diams:
            self._add_circle(obj, center, diam / 2)

            d_i = d_o
            d_o = diam
            if d_i is not None and d_o is not None:
                rings.append(Ring(d_i, d_o))

        return rings

    @staticmethod
    def _add_circle(obj, center, radius):
        if isinstance(obj, Axes):
            obj.add_patch(plt.Circle(center, radius, color='black', fill=False))
        elif isinstance(obj, Modelspace):
            obj.add_circle(center, radius)


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
        res = super().total_cut_length
        return res  # TODO add arc lenghs of inner cuts

    def plot(self, ax=None, x0=None):
        if ax is None:
            _, ax = plt.subplots()

        self._draw(ax, x0)

    def add_to_dxf(self, msp: Modelspace, x0=None, layer: str = 'background') -> List[Ring]:
        return self._draw(msp, x0)

    def _draw(self, obj, x0) -> List[Ring]:
        if x0 is None:
            x0 = np.zeros([2])
        center = np.asarray([self.diameter / 2] * 2) + x0

        rings = super()._draw(obj, x0)
        rings = [Ring(diam_inner=self.rings[0][1], diam_outer=self.rings[0][2], has_socket=True)] + rings

        # plot inner socket ring
        self._add_circle(obj, center, self.socket_diam / 2)

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
                self._add_arc(obj, center, d, theta1, theta2)

        # plot struts
        for x_sign in {-1, 1}:
            for y_sign in {-1, 1}:
                v_sign = np.asarray([x_sign, y_sign])
                self._add_line(obj, center + v_sign * p_strut_inner, center + v_sign * p_strut_outer)

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
                    self._add_arc(obj, center + p_corner * v_sign, diam, theta1, theta2)

        return rings

    def _get_points_inner(self):
        r_inner = self.socket_diam / 2 + self.socket_ring_width
        y_strut = self.strut_width / 2
        r_corner = self.socket_ring_corner_diam / 2
        y_corner = y_strut + r_corner
        p_corner = np.asarray([np.sqrt((r_inner + r_corner) ** 2 - y_corner ** 2), y_corner])
        p_strut = np.asarray([p_corner[0], y_strut])
        p_inner = p_corner / np.linalg.norm(p_corner) * r_inner

        return p_strut, p_inner, p_corner

    def _get_points_outer(self):
        r_outer = self.rings[0][1] / 2
        y_strut = self.strut_width / 2
        r_corner = self.outer_ring_corner_diam / 2
        y_corner = y_strut + r_corner
        p_corner = np.asarray([np.sqrt((r_outer - r_corner) ** 2 - y_corner ** 2), y_corner])
        p_strut = np.asarray([p_corner[0], y_strut])
        p_outer = p_corner / np.linalg.norm(p_corner) * r_outer

        return p_strut, p_outer, p_corner

    @staticmethod
    def _add_arc(obj, center, diameter, start_angle, end_angle):
        if isinstance(obj, Axes):
            obj.add_patch(patches.Arc(center, diameter, diameter, angle=0.0, theta1=start_angle, theta2=end_angle))
        elif isinstance(obj, Modelspace):
            obj.add_arc(center, diameter / 2, start_angle, end_angle)

    @staticmethod
    def _add_line(obj, start, end):
        if isinstance(obj, Axes):
            obj.add_line(plt.Line2D([start[0], end[0]], [start[1], end[1]], color='black'))
        else:
            obj.add_line(start, end)


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
    rings_old = rings
    rings = np.zeros_like(rings_old, shape=[rings.shape[0], rings.shape[1] + 1])
    rings[:, :5] = rings_old

    strut_rings = []
    new_rings = None
    for lamp_idx, lamp in enumerate(lamps):
        lamp_rings = rings[rings[:, 3] == lamp_idx]
        lamp_rings = lamp_rings[lamp_rings[:, 4].argsort()]
        lamp_rings = np.flip(lamp_rings, axis=0)
        for idx in lamp.socket_layer_indices:
            lamp_rings[idx, -1] = 1
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
        cut_area.add_rings(other.rings)
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

    @property
    def lims(self) -> np.ndarray:
        if self.work_area[0] == np.infty:
            return self.size
        else:
            return self.work_area

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
            self._top_left_corners.append(np.asarray([x0 + ca.padding, y0 + ca.padding]))
            row_max_diam = max(row_max_diam, ca.job_diameter)
            x0 += ca.job_diameter

        return cas

    def plot(self, ax):
        ax.set_aspect('equal', 'box')

        ax.set_xlim((0, self.lims[0]))
        ax.set_ylim((0, self.lims[1]))

        for cut_area, corner in zip(self._cut_areas, self._top_left_corners):
            cut_area.plot(ax=ax, x0=corner)

    def save_dxf(self, zf: Optional[ZipFile] = None, filename: Optional[str] = None) -> List[Ring]:
        doc = ezdxf.new()
        doc.units = units.MM
        msp = doc.modelspace()

        rings = []

        for cut_area, corner in zip(self._cut_areas, self._top_left_corners):
            rings += cut_area.add_to_dxf(msp=msp, x0=corner)

        if zf is not None and filename is not None:
            with zf.open(filename, 'w') as f:
                doc.write(stream=TextIOWrapper(f), fmt='asc')

        return rings


def get_jobs(cut_areas: List[CutArea], work_area: Optional[np.ndarray] = None) -> List[Job]:
    if work_area is None:
        work_area = np.asarray([np.infty] * 2)

    if max([cut_area.diameter for cut_area in cut_areas]) > work_area[:].min():
        raise ValueError("Work area is too small!")

    cas = cut_areas[:]
    jobs = []
    while cas:
        job = Job(work_area)
        cas = job.assign_cut_areas(cas)
        jobs.append(job)

    return jobs


def generate_drawing(jobs: List[Job], zip_path: str = None, plot_aspect_ratio: float = 16 / 9) -> List[Ring]:
    njobs = len(jobs)
    nrows = int(np.floor(np.sqrt(njobs / plot_aspect_ratio)))
    ncols = int(np.ceil(njobs / nrows))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for idx in range(njobs, nrows * ncols):
        fig.delaxes(axs[np.unravel_index(idx, [nrows, ncols])])

    fig.suptitle("technical drawing")
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    rings = []

    zf = ZipFile(zip_path, 'w') if zip_path is not None else None
    for idx, job in enumerate(jobs):
        ax = axs[np.unravel_index(idx, [nrows, ncols])]
        ax.title.set_text(f'Job #{idx + 1}')
        job.plot(ax)
        rings += job.save_dxf(zf, f"job_{idx + 1}.dxf")

    return rings


def assign_rings(rings: List[Ring], all_rings: np.ndarray) -> List[Ring]:
    all_rings = all_rings.copy()

    assigned_rings = []
    while rings and all_rings.shape[0]:
        ring = get_next_ring(rings, all_rings)
        all_rings = assign_ring(ring, all_rings)
        if ring.lamp_idx is None:
            break
        rings.remove(ring)
        assigned_rings.append(ring)

    return assigned_rings


def get_next_ring(rings: List[Ring], all_rings: np.ndarray) -> Ring:
    rings = sorted(rings, key=lambda ring: (np.min(np.abs(ring.diam_outer - all_rings[:, 2])), np.min(np.abs(ring.diam_inner - all_rings[:, 1]))))
    return rings[0]


def assign_ring(ring: Ring, rings: np.ndarray) -> np.ndarray:
    candidates = rings.astype(float)
    candidates[rings[:, 2] != ring.diam_outer, 1] = np.infty
    candidates[rings[:, -1] != int(ring.has_socket), 1] = np.infty
    deviations = np.abs(candidates[:, 1] - ring.diam_inner)
    if min(deviations) >= np.infty:
        return rings
    idx = np.argmin(deviations)
    ring.lamp_idx = rings[idx, 3]
    ring.z = rings[idx, 0]
    ring.z_idx = rings[idx, 4]
    return np.delete(rings, idx, axis=0)


def main(visualize=False):
    lamps = [
        LampParams(
            sphere_diameter=270,
            layer_thickness=6.7,
            ring_width=15,
            sphere_opening_top=95,
            sphere_opening_bottom=115,
            socket_diam=40,
            socket_ring_width=25,
            strut_width=40,
            outer_ring_corner_diam=25,
            socket_ring_corner_diam=10,
            socket_layer_indices=[5, 6],
            bulb_diameter=95,
            bulb_length=135
        ),
        LampParams(
            sphere_diameter=360,
            layer_thickness=6.7,
            ring_width=16,
            sphere_opening_top=115,
            sphere_opening_bottom=170,
            socket_diam=40,
            socket_ring_width=25,
            strut_width=40,
            outer_ring_corner_diam=25,
            socket_ring_corner_diam=10,
            socket_layer_indices=[10, 11],
            bulb_diameter=95,
            bulb_length=135
        ),
        LampParams(
            sphere_diameter=450,
            layer_thickness=6.7,
            ring_width=20,
            sphere_opening_top=150,
            sphere_opening_bottom=235,
            socket_diam=40,
            socket_ring_width=25,
            strut_width=40,
            outer_ring_corner_diam=25,
            socket_ring_corner_diam=10,
            socket_layer_indices=[13, 14],
            bulb_diameter=125,
            bulb_length=175
        )
    ]
    tolerance = 4
    padding = 10
    work_area = np.asarray([940, 565])

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
                    outer_ring=ring,
                    outer_ring_corner_diam=lamp.outer_ring_corner_diam,
                    socket_ring_corner_diam=lamp.socket_ring_corner_diam,
                    tolerance=tolerance,
                    padding=padding
                )
            )
    cut_areas = merge_cut_areas(cut_areas)
    jobs = get_jobs(cut_areas, work_area)

    all_rings = np.stack([ring for cut_area in cut_areas for ring in cut_area.cut_rings], axis=0)

    for cut_area in cut_areas:
        print(cut_area.cut_diams)

    for lamp_idx, lamp in enumerate(lamps):
        print(f"Lamp #{lamp_idx + 1} has {len(all_rings[all_rings[:, 3] == lamp_idx])} rings")

    print(f"#rings (total):     {len(all_rings)}")
    print(f"#cuts (total):      {sum([len(cut_area.cut_diams) for cut_area in cut_areas])}")
    print(f"#cut areas:         {len(cut_areas)}")
    print(f"#jobs:              {len(jobs)}")
    print(f"total cut length:   {sum([cut_area.total_cut_length for cut_area in cut_areas]) * 1e-3:.02f}m")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    rings = generate_drawing(jobs, zip_path=os.path.join("output", f"{timestr}_jobs.zip"))
    rings = assign_rings(rings, all_rings)

    if visualize:
        zmin = min([ring.z for ring in rings]) - 25
        zmax = max([ring.z for ring in rings]) + 25
        xmax = max([ring.diam_outer / 2 for ring in rings]) + 25
        xmin = -xmax
        fig, ax = plt.subplots(1, 3)
        plt.suptitle("cardboard lamps")
        for lamp_idx, lamp in enumerate(lamps):
            ax[lamp_idx].set_aspect('equal', 'box')
            ax[lamp_idx].set_xlim(xmin, xmax)
            ax[lamp_idx].set_ylim(zmin, zmax)
            lamp_rings = [ring for ring in rings if ring.lamp_idx == lamp_idx]
            for ring in lamp_rings:
                if not ring.has_socket:
                    for sign in {-1, 1}:
                        ax[lamp_idx].plot(
                            [sign * ring.diam_inner / 2, sign * ring.diam_outer / 2], [ring.z, ring.z],
                            linewidth=lamp.layer_thickness,
                            color='brown'
                        )
                else:
                    ax[lamp_idx].plot(
                        [-ring.diam_outer / 2, ring.diam_outer / 2], [ring.z, ring.z],
                        linewidth=lamp.layer_thickness,
                        color='brown'
                    )

            z_strut = min([ring.z for ring in lamp_rings if ring.has_socket]) - lamp.layer_thickness / 2
            for sign in {-1, 1}:
                ax[lamp_idx].plot(
                    [sign * 27 / 2, sign * 27 / 2], [z_strut, z_strut - lamp.socket_length],
                    linewidth=1,
                    color='black'
                )
            center = (0.0, z_strut - lamp.socket_length - lamp.bulb_diameter / 2)
            circle = plt.Circle(center, lamp.bulb_diameter / 2, color='yellow', fill=True)
            ax[lamp_idx].add_patch(circle)

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
    main(visualize=True)
