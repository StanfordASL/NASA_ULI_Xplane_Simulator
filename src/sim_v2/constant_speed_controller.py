import jax
import jax.numpy as jnp
import numpy as np

import gps
import ilqr

# `NamedTuple`s are used (more accurately, abused) in this notebook to minimize dependencies;
# much better JAX-compatible choices to fit the archetype of "parameterized function" would be
# `flax.struct.dataclass` or `equinox.Module`.
from typing import Callable, NamedTuple


def frenet_coordinates(pose, waypoints):

    def pose_segment_frenet(pose, vw):
        p, q = pose[:2], pose[-1]
        v, w = vw

        l = jnp.linalg.norm(w - v)
        n = (w - v) / l
        s = jnp.clip((p - v) @ n, 0, l)
        t = (p - v) @ jnp.array([-n[1], n[0]])
        h = jnp.mod(q - jnp.arctan2(n[1], n[0]) + np.pi, 2 * np.pi) - np.pi
        c = v + s * n
        d2 = jnp.sum(jnp.square(p - c))
        return (s, t, h), d2, l

    (s, t, h), d2, l = jax.vmap(pose_segment_frenet, in_axes=(None, 0))(pose, (waypoints[:-1], waypoints[1:]))
    s = s + jnp.cumsum(l) - l
    i = jnp.argmin(d2 - 1e-2 * np.arange(len(d2)))  # Break ties in favor of later segments.
    return s[i], t[i], h[i]


class DubinsCarDynamics(NamedTuple):
    speed: float = 5.0

    def __call__(self, state, control):
        e, n, q = state
        k, = control
        v = self.speed
        return jnp.array([
            v * jnp.cos(q),
            v * jnp.sin(q),
            v * k,
        ])


class WaypointTrackingRunningCost(NamedTuple):
    waypoints_en: np.array
    soft_lateral_error_limit: float = 5.0
    soft_curvature_limit: float = 0.1

    def __call__(self, state, control, step):
        return sum(self.components(state, control, step))

    def components(self, state, control, step):
        k, = control

        s, t, h = frenet_coordinates(state, self.waypoints_en)
        lateral_error_limit = 100 * jnp.maximum(jnp.abs(t) - self.soft_lateral_error_limit, 0)**2
        curvature_cost = k**2
        curvature_limit = 100 * jnp.maximum(jnp.abs(k) - self.soft_curvature_limit, 0)**2

        return (lateral_error_limit, curvature_cost, curvature_limit)


class WaypointTrackingTerminalCost(NamedTuple):
    waypoints_en: np.array

    def __call__(self, state):
        s, t, h = frenet_coordinates(state, self.waypoints_en)
        return 10 * (t**2 + 10 * h**2) - s


class ConstantSpeedController:

    def __init__(self, client):
        self.client = client

        llaxyz = [
            x[0] for x in client.getDREFs([
                "sim/flightmodel/position/" + x for x in [
                    "latitude",
                    "longitude",
                    "elevation",
                    "local_x",
                    "local_y",
                    "local_z",
                ]
            ])
        ]
        self.coordinate_converter = gps.CoordinateConverter.from_lla_and_xyz(llaxyz[:3], llaxyz[-3:], jnp)
        self.ll2en = jax.jit(self.coordinate_converter.ll2en)
        self.batch_ll2en = jax.jit(jax.vmap(self.coordinate_converter.ll2en))

        self.prev_state = self.prev_control = self.prev_t = None

    def get_control(self, belief, waypoints):
        if self.prev_t is not None:
            curr_t = belief["t"]
            curr_state = ilqr.RK4Integrator(DubinsCarDynamics(), min(curr_t - self.prev_t, 0.2))(self.prev_state,
                                                                                                 self.prev_control, 0)
            e, n, q = np.array(curr_state)
            print((e, n, q), (e, -n, np.mod(90 - q * 180 / np.pi, 360)))
            self.client.sendDREFs(["sim/flightmodel/position/" + x for x in [
                "local_x",
                "local_z",
            ]], (e, -n))
            # self.client.sendDREF("sim/flightmodel/position/psi", np.mod(90 - q * 180 / np.pi, 360))  # bugged?
            quat = np.array([
                np.cos(np.pi / 4 - q / 2), *self.client.getDREF("sim/flightmodel/position/q")[1:3],
                np.sin(np.pi / 4 - q / 2)
            ])
            self.client.sendDREF("sim/flightmodel/position/q", quat / np.linalg.norm(quat))

        # e, n = self.ll2en((belief["Latitude"], belief["Longitude"]))  # TODO: fix lingering coordinate mismatch
        e = self.client.getDREF("sim/flightmodel/position/local_x")[0]
        n = -self.client.getDREF("sim/flightmodel/position/local_z")[0]
        q = (90 - self.client.getDREF("sim/flightmodel/position/psi")[0]) * np.pi / 180

        state = np.array([e, n, q])
        waypoints_en = self.batch_ll2en(waypoints.to_numpy())

        dynamics = ilqr.RK4Integrator(DubinsCarDynamics(), 0.2)
        cost = ilqr.TotalCost(WaypointTrackingRunningCost(waypoints_en), WaypointTrackingTerminalCost(waypoints_en))

        solution = ilqr.iterative_linear_quadratic_regulator(dynamics, cost, state, np.zeros((20, 1)))
        states, controls = solution["optimal_trajectory"]
        self.prev_state = state
        self.prev_control = controls[0]
        self.prev_t = belief["t"]

        self.client.sendCTRL([-998])
