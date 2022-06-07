import numpy as np
import scipy


def make_smooth_spline(waypoints, ds=1, smoothing_factor=10):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=-1))]
    dense_s = np.linspace(0, s[-1], round(s[-1] / ds))
    (t, c, k), _ = scipy.interpolate.splprep([np.interp(dense_s, s, waypoints_dim) for waypoints_dim in waypoints.T],
                                             s=smoothing_factor)
    return scipy.interpolate.BSpline(t, np.array(c).T, k)


def compute_spline_arc_length(spline, t):
    t = np.r_[spline.t[0], t]
    return np.cumsum(
        scipy.integrate.quad_vec(
            lambda a: np.linalg.norm(spline((1 - a) * t[:-1] + a * t[1:], 1), axis=-1),
            0,
            1,
        )[0] * np.diff(t))


def reparameterize_spline_by_arc_length(spline, ds=1):
    s = compute_spline_arc_length(spline, spline.t)
    t = np.interp(np.linspace(0, s[-1], round(s[-1] / ds) + 1), s, spline.t)
    for _ in range(10):
        s = compute_spline_arc_length(spline, t)
        target_s = np.linspace(0, s[-1], t.size)
        if np.max(np.abs(target_s - s)) < 1e-6:
            break
        t = t + (target_s - s) / np.linalg.norm(spline(t, 1), axis=-1)
    return scipy.interpolate.make_interp_spline(s, spline(t))


class Trajectory:

    def __init__(self, waypoints_en, ds=10, smoothing_factor=10):
        self.waypoints_en = waypoints_en
        self.en = reparameterize_spline_by_arc_length(make_smooth_spline(self.waypoints_en, ds, smoothing_factor), ds)
        self.en_knot_points = self.en(self.en.t)
        self.e = scipy.interpolate.PPoly.from_spline((self.en.t, self.en.c[:, 0], self.en.k))
        self.n = scipy.interpolate.PPoly.from_spline((self.en.t, self.en.c[:, 1], self.en.k))
        self.ed, self.edd = self.e.derivative(), self.e.derivative(2)
        self.nd, self.ndd = self.n.derivative(), self.n.derivative(2)
        self.en_dot_end_coefficients = np.array([
            sum(self.e.c[i - j] * self.ed.c[j] + self.n.c[i - j] * self.nd.c[j] if 0 <= i - j <= self.en.k else 0
                for j in range(self.en.k))
            for i in range(2 * self.en.k)
        ])
        self.end_coefficients = np.pad(np.stack([self.ed.c, self.nd.c], -1), ((self.en.k, 0), (0, 0), (0, 0)))

    @classmethod
    def from_latitudes_and_longitudes(cls, coordinate_converter, latitudes, longitudes, ds=10, smoothing_factor=10):
        waypoints_ll = np.c_[latitudes, longitudes]
        trajectory = cls(np.array([coordinate_converter.ll2en(ll) for ll in waypoints_ll]), ds, smoothing_factor)
        trajectory.coordinate_converter = coordinate_converter
        trajectory.waypoints_ll = waypoints_ll
        return trajectory

    def __call__(self, s):
        e, ed, edd = self.e(s), self.ed(s), self.edd(s)
        n, nd, ndd = self.n(s), self.nd(s), self.ndd(s)
        q = np.arctan2(nd, ed)
        k = (ed * ndd - nd * edd) / np.hypot(ed, nd)**3
        return np.stack([e, n, q, k], -1)

    def find_closest_point_s(self, point, extrapolate=False):
        i = np.argmin(np.sum(np.square(self.en_knot_points - point), -1))
        j, k = max(i - 2, 0), min(i + 3, self.en_knot_points.size - 1)
        s = scipy.interpolate.PPoly.construct_fast(
            self.en_dot_end_coefficients[:, j:k - 1] - self.end_coefficients[:, j:k - 1, :] @ point,
            self.e.x[j:k],
            extrapolate,
        ).roots()
        if s.size == 0:
            return self.e.x[i]
        return min(zip(np.sum(np.square(self.en(s) - point), -1), s))[1]

    def frenet_coordinates(self, point, extrapolate=False, return_reference_point=False):
        s_ref = self.find_closest_point_s(point, extrapolate)
        e_ref, n_ref, q_ref, k_ref = enqk_ref = self(s_ref)
        v = point - enqk_ref[:2]
        d = np.linalg.norm(v) * np.sign(v @ np.array([-np.sin(q_ref), np.cos(q_ref)]))
        if return_reference_point:
            return np.array([s_ref, d]), enqk_ref
        return np.array([s_ref, d])
