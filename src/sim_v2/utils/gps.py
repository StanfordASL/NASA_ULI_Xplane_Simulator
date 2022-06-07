# See https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_geodetic_coordinates
import jax
import jax.numpy as jnp
import numpy as np

from . import sim_data

# CONSTANTS
D2R = np.pi / 180
R2D = 180 / np.pi
a = 6378137  # ellipsoid's semi-major axis
b = 6356752.3  # ellipsoid's semi-minor axis
e2 = 1 - b * b / (a * a)  # ellipsoid's eccentricity^2
ep2 = e2 / (1 - e2)  # used in ecef2lla
E2 = e2 * a * a  # used in ecef2lla


def lla2ecef(lla, np=np):
    clat = np.cos(lla[0] * D2R)
    clon = np.cos(lla[1] * D2R)
    slat = np.sin(lla[0] * D2R)
    slon = np.sin(lla[1] * D2R)
    r0 = a / (np.sqrt(1.0 - e2 * slat * slat))
    return np.array([
        (lla[2] + r0) * clat * clon,
        (lla[2] + r0) * clat * slon,
        (lla[2] + r0 * (1.0 - e2)) * slat,
    ])


def ecef2lla(ecef, np=np):
    X = ecef[0]
    Y = ecef[1]
    Z = ecef[2]

    r = np.hypot(X, Y)
    F = 54 * b * b * Z * Z
    G = r * r + (1 - e2) * Z * Z - e2 * E2
    C = e2 * e2 * F * r * r / (G * G * G)
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
    T = S + 1 / S + 1
    P = F / (3 * T * T * G * G)
    Q = np.sqrt(1 + 2 * e2 * e2 * P)
    r0 = -P * e2 * r / (1 + Q) + np.sqrt((1 + 1 / Q) * a * a / 2 - P * (1 - e2) * Z * Z / (Q * (1 + Q)) - P * r * r / 2)
    U = np.hypot(r - e2 * r0, Z)
    V = np.hypot(r - e2 * r0, Z * b / a)
    Z0 = b * b * Z / (a * V)
    return np.array([np.arctan((Z + ep2 * Z0) / r) * R2D, np.arctan2(Y, X) * R2D, U * (1 - b * b / (a * V))])


def R_ecef2enu(llaRef, np=np):
    clatRef = np.cos(llaRef[0] * D2R)
    clonRef = np.cos(llaRef[1] * D2R)
    slatRef = np.sin(llaRef[0] * D2R)
    slonRef = np.sin(llaRef[1] * D2R)
    return np.array([[-slonRef, clonRef, 0], [-slatRef * clonRef, -slatRef * slonRef, clatRef],
                     [clatRef * clonRef, clatRef * slonRef, slatRef]])


def lla2enu(llaRef, lla, np=np):
    ecefRef, R = lla2ecef(llaRef, np), R_ecef2enu(llaRef, np)
    ecef = lla2ecef(lla, np)
    enu = R @ (ecef - ecefRef)
    return enu


def enu2lla(llaRef, enu, np=np):
    ecefRef, R = lla2ecef(llaRef, np), R_ecef2enu(llaRef, np)
    ecef = np.linalg.solve(R, enu) + ecefRef
    lla = ecef2lla(ecef, np)
    return lla


def lla2xyz(llaRef, lla, np=np):
    e, n, u = lla2enu(llaRef, lla, np)
    return np.array([e, u, -n])


def xyz2lla(llaRef, xyz, np=np):
    x, y, z = xyz
    return enu2lla(llaRef, (x, -z, y), np)


def fit_llaRef(lla, xyz):

    @jax.jit
    def xyz_residual(llaRef):
        return jnp.array(xyz) - lla2xyz(llaRef, lla, jnp)

    @jax.jit
    def newton_step(llaRef):
        return llaRef - jnp.linalg.solve(jax.jacobian(xyz_residual)(llaRef), xyz_residual(llaRef))

    lat, lon, _ = lla
    inc = np.arange(-3, 4) / 2
    lat_ref, lon_ref, _ = min([(lat_ref, lon_ref, np.linalg.norm(xyz_residual((lat_ref, lon_ref, 0))))
                               for lat_ref in round(lat) + inc
                               for lon_ref in round(lon) + inc],
                              key=lambda x: x[-1])
    llaRef = jnp.array([lat_ref, lon_ref, 0])
    for _ in range(10):
        llaRef = newton_step(llaRef)
    return llaRef, xyz_residual(llaRef)


class CoordinateConverter:

    def __init__(self, llaRef, np=np):
        self.llaRef = llaRef
        self.np = np

    @classmethod
    def from_lla_and_xyz(cls, lla, xyz, np=np):
        llaRef, xyz_residual = fit_llaRef(lla, xyz)
        if np.max(np.abs(xyz_residual)) > 10:
            raise ValueError(f"Couldn't find a good `llaRef` fit for `lla` {lla} and `xyz` {xyz}.")
        converter = cls(np.array(llaRef), np)
        converter.xyz_residual = np.array(xyz_residual)
        return converter

    @classmethod
    def from_sim(cls, client, np=np):
        llaxyz = sim_data.CoordinateConversionDREFs.get_from_sim(client)
        return cls.from_lla_and_xyz((llaxyz.latitude, llaxyz.longitude, llaxyz.elevation),
                                    (llaxyz.local_x, llaxyz.local_y, llaxyz.local_z), np)

    def lla2enu(self, lla):
        return lla2enu(self.llaRef, lla, self.np)

    def enu2lla(self, enu):
        return enu2lla(self.llaRef, enu, self.np)

    def lla2xyz(self, lla):
        return lla2xyz(self.llaRef, lla, self.np)

    def xyz2lla(self, xyz):
        return xyz2lla(self.llaRef, xyz, self.np)

    def ll2en(self, ll):
        return lla2enu(self.llaRef, (*ll, self.llaRef[-1]), self.np)[:2]

    def en2ll(self, en):
        return enu2lla(self.llaRef, (*en, 0), self.np)[:2]
