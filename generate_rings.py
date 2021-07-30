import matplotlib.pyplot as plt
import numpy as np

def compute_rings(
    sphere_diameter: float,
    layer_thickness: float,
    ring_width: float,
    sphere_opening_top: float,
    sphere_opening_bottom: float
) -> np.ndarray:
    # start with center ring
    r_o = sphere_diameter / 2
    r_i = r_o - ring_width
    res = [[0, r_i, r_o]]

    # top rings
    z = 0
    while r_i > sphere_opening_top / 2:
        z += layer_thickness
        r_o = np.sqrt((sphere_diameter / 2) ** 2 - z**2)
        r_i = r_o - ring_width
        res = [[z, r_i, r_o]] + res
    
    # bottom rings
    z = 0
    r_i = sphere_diameter / 2 - ring_width
    while r_i > sphere_opening_bottom / 2:
        z -= layer_thickness
        r_o = np.sqrt((sphere_diameter / 2) ** 2 - z**2)
        r_i = r_o - ring_width
        res += [[z, r_i, r_o]]

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

    return res


class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()


def main():
    sphere_diameter = 450
    layer_thickness = 8
    ring_width = 15
    sphere_opening_top = 150
    sphere_opening_bottom = 250

    rings = compute_rings(
        sphere_diameter,
        layer_thickness,
        ring_width,
        sphere_opening_top,
        sphere_opening_bottom
    )

    print(rings[rings[:, 2].argsort()])
    print(f"#rings: {len(rings)}")

    fig, ax = plt.subplots(1, 1)
    plt.suptitle("cardboard lamp")
    ax.axis('equal')
    for z, r_i, r_o in rings:
        data_linewidth_plot([r_i, r_o], [z, z], ax=ax, linewidth=layer_thickness, color='brown')
        data_linewidth_plot([-r_i, -r_o], [z, z], ax=ax, linewidth=layer_thickness, color='brown')
    plt.show()


if __name__ == "__main__":
    main()
