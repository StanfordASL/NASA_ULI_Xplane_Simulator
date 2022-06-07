import dataclasses

import numpy as np

from typing import NamedTuple

RAD_PER_DEG = np.pi / 180
DEG_PER_RAD = 180 / np.pi
wrap_pi = lambda x, np=np: np.mod(x + np.pi, 2 * np.pi) - np.pi


@dataclasses.dataclass
class DREFs:

    @staticmethod
    def DREF_prefix():
        return ""

    @classmethod
    def dref_names(cls):
        return [cls.DREF_prefix() + k for k in cls.__dataclass_fields__.keys()]

    @classmethod
    def get_from_sim(cls, client, include_timestamp=False):

        def unpack_scalars(drefs):
            return cls(*[(x if v.type is tuple else x[0]) for x, v in zip(drefs, cls.__dataclass_fields__.values())])

        if include_timestamp:
            drefs = client.getDREFs((["sim/time/local_time_sec"] if include_timestamp else []) + cls.dref_names())
            return drefs[0][0], unpack_scalars(drefs[1:])
        return unpack_scalars(client.getDREFs(cls.dref_names()))

    def send_to_sim(self, client):
        client.sendDREFs(self.dref_names(), dataclasses.astuple(self))


@dataclasses.dataclass
class SimData:

    @classmethod
    def get_from_sim(cls, client, include_timestamp=False):
        sim_data = cls(*[v.type.get_from_sim(client) for v in cls.__dataclass_fields__.values()])
        if include_timestamp:
            return client.getDREF("sim/time/local_time_sec")[0], sim_data
        return sim_data


@dataclasses.dataclass
class StateDREFs(DREFs):
    local_x: float
    local_y: float
    local_z: float
    lat_ref: float
    lon_ref: float
    latitude: float
    longitude: float
    elevation: float
    theta: float
    phi: float
    psi: float
    magpsi: float
    true_theta: float
    true_phi: float
    true_psi: float
    mag_psi: float
    local_vx: float
    local_vy: float
    local_vz: float
    local_ax: float
    local_ay: float
    local_az: float
    alpha: float
    beta: float
    vpath: float
    hpath: float
    groundspeed: float
    indicated_airspeed: float
    indicated_airspeed2: float
    true_airspeed: float
    magnetic_variation: float
    M: float
    N: float
    L: float
    P: float
    Q: float
    R: float
    P_dot: float
    Q_dot: float
    R_dot: float
    Prad: float
    Qrad: float
    Rrad: float
    q: tuple
    vh_ind: float
    vh_ind_fpm: float
    vh_ind_fpm2: float
    y_agl: float

    @staticmethod
    def DREF_prefix():
        return "sim/flightmodel/position/"


@dataclasses.dataclass
class MutableStateDREFs(DREFs):
    local_x: float
    local_y: float
    local_z: float
    theta: float
    phi: float
    psi: float
    local_vx: float
    local_vy: float
    local_vz: float
    local_ax: float
    local_ay: float
    local_az: float
    indicated_airspeed: float
    indicated_airspeed2: float
    P: float
    Q: float
    R: float
    Prad: float
    Qrad: float
    Rrad: float
    q: tuple
    vh_ind_fpm: float
    vh_ind_fpm2: float

    @staticmethod
    def DREF_prefix():
        return "sim/flightmodel/position/"


@dataclasses.dataclass
class CoordinateConversionDREFs(DREFs):
    latitude: float
    longitude: float
    elevation: float
    local_x: float
    local_y: float
    local_z: float

    @staticmethod
    def DREF_prefix():
        return "sim/flightmodel/position/"


@dataclasses.dataclass
class TaxiStateDREFs(DREFs):
    local_x: float
    local_z: float
    psi: float
    local_vx: float
    local_vz: float
    R: float
    local_ax: float
    local_az: float
    R_dot: float

    @staticmethod
    def DREF_prefix():
        return "sim/flightmodel/position/"

    def to_body_frame(self):
        e, n, q = self.local_x, -self.local_z, wrap_pi((90 - self.psi) * RAD_PER_DEG)
        cq, sq = np.cos(q), np.sin(q)
        v_x = self.local_vx * cq - self.local_vz * sq
        v_y = -self.local_vx * sq - self.local_vz * cq
        r = -self.R * RAD_PER_DEG
        v_x_dot = self.local_ax * cq - self.local_az * sq + v_y * r
        v_y_dot = -self.local_ax * sq - self.local_az * cq - v_x * r
        r_dot = -self.R_dot * RAD_PER_DEG
        return TaxiState(e, n, q, v_x, v_y, r, v_x_dot, v_y_dot, r_dot)


class TaxiState(NamedTuple):
    e: float
    n: float
    q: float
    v_x: float
    v_y: float
    r: float
    v_x_dot: float
    v_y_dot: float
    r_dot: float

    @classmethod
    def get_from_sim(cls, client):
        return TaxiStateDREFs.get_from_sim(client).to_body_frame()


class XPlaneControl(NamedTuple):
    parkbrake: float
    elevator: float
    aileron: float
    rudder: float
    throttle: float
    gear: float = 1.0
    flaps: float = 0.0
    speedbrakes: float = 0.0

    def to_taxi_control(self):
        return TaxiControl(self.parkbrake, -self.aileron, self.throttle)

    @classmethod
    def get_from_sim(cls, client):
        return cls(client.getDREF("sim/flightmodel/controls/parkbrake")[0], *client.getCTRL())

    def send_to_sim(self, client):
        client.sendDREF("sim/flightmodel/controls/parkbrake", self.parkbrake)
        client.sendCTRL(self[1:])


class TaxiControl(NamedTuple):
    parkbrake: float
    steering: float
    throttle: float

    def to_xplane_control(self):
        aileron = -np.clip(self.steering, -1, 1)
        return XPlaneControl(self.parkbrake, 0, aileron, aileron / 5, self.throttle)

    @classmethod
    def get_from_sim(cls, client):
        return XPlaneControl.get_from_sim(client).to_taxi_control()

    def send_to_sim(self, client):
        self.to_xplane_control().send_to_sim(client)
