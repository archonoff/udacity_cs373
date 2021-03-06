import numpy as np
from math import *


class KalmanFilterBase:
    state_size = NotImplemented
    measurement_size = NotImplemented

    def __init__(self, measurement=None, R_multiplier=100, R=None, Q_multiplier=.001, Q=None, P_multiplier=1000):
        if measurement is None:
            x_measurement, y_measurement = (0, 0)
        else:
            x_measurement, y_measurement = measurement

        self.X = self._init_X(x_measurement, y_measurement)
        self.I = np.matrix(np.identity(self.state_size))
        self.H = self._get_H(self.X)
        self.P = self._get_P(self.X, P_multiplier=P_multiplier)
        if R is not None:
            self.R = R
        else:
            self.R = self._get_R(self.X, R_multiplier=R_multiplier)

        self.Q = Q

        self.R_multiplier = R_multiplier
        self.Q_multiplier = Q_multiplier
        self.P_multiplier = P_multiplier

    def get_position(self):
        """Gets last estimated position"""
        xy_estimate = self.X[0, 0], self.X[1, 0]
        return xy_estimate

    def _get_P(self, X, P_multiplier, dt=None):
        P = self.I * P_multiplier
        return P

    def _get_R(self, X, R_multiplier, dt=None):
        R = np.matrix(np.identity(self.measurement_size)) * R_multiplier
        return R

    def _init_X(self, *args, **kwargs):
        raise NotImplementedError

    def _get_F(self, X, dt=None):
        raise NotImplementedError

    def _get_H(self, X, dt=None):
        raise NotImplementedError

    def step(self, measurement, dt=1):
        raise NotImplementedError

    def get_prediction(self, steps=1, dt=1):
        raise NotImplementedError


class KalmanFilter(KalmanFilterBase):
    def __init__(self, *args, **kwargs):
        self.state_size = 6
        self.measurement_size = 2

        super().__init__(*args, **kwargs)

    def _init_X(self, x_measurement, y_measurement):
        return np.matrix([x_measurement, y_measurement, 0, 0, 0, 0]).T

    def _get_F(self, X, dt=None):
        F = np.matrix([
            [1, 0, dt, 0, dt**2, 0],
            [0, 1, 0, dt, 0, dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        return F

    def _get_H(self, X, dt=None):
        H = np.matrix([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0]])
        return H

    def step(self, measurement, dt=1):
        F = self._get_F(self.X, dt)
        if self.Q is None:
            Q = F * F.T * self.Q_multiplier
        else:
            Q = self.Q

        self.X = F * self.X
        self.P = F * self.P * F.T + Q

        Z = np.matrix(measurement).T
        Y = Z - self.H * self.X
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.pinv(S)
        self.X = self.X + K * Y
        self.P = (self.I - K * self.H) * self.P

    def get_prediction(self, steps=1, dt=1):
        """Gets prediction or robot position after given number of steps"""
        X = self.X

        for step in range(steps):
            F = self._get_F(X, dt)
            X = F * X

        xy_estimate = X[0, 0], X[1, 0]

        return xy_estimate


class ExtendedKalmanFilter(KalmanFilterBase):
    prev_x = None
    prev_y = None
    prev_b = None

    def __init__(self, *args, **kwargs):
        self.state_size = 5
        self.measurement_size = 2

        super().__init__(*args, **kwargs)

    def _unpack_X(self, X):
        x = X[0, 0]
        y = X[1, 0]
        v = X[2, 0]
        b = X[3, 0]
        w = X[4, 0]
        return x, y, v, b, w

    def _f(self, X, dt):
        x, y, v, b, w = self._unpack_X(X)
        x, y, b = x + sin(b) * v * dt, y + cos(b) * v * dt, b + w * dt
        return np.matrix((x, y, v, b, w)).T

    def _init_X(self, x_measurement, y_measurement):
        return np.matrix([x_measurement, y_measurement, 0.1, 0.1, 0.1]).T

    def _get_F(self, X, dt=None):
        x, y, v, b, w = self._unpack_X(X)
        F = np.matrix(np.identity(5))

        F[0, 2] = sin(b + w * dt) * dt
        F[0, 3] = cos(b + w * dt) * v * dt
        F[0, 4] = cos(b + w * dt) * v * dt**2

        F[1, 2] = cos(b + w * dt) * dt
        F[1, 3] = -sin(b + w * dt) * v * dt
        F[1, 4] = -sin(b + w * dt) * v * dt**2

        F[3, 4] = dt

        return F

    def _get_H(self, X, dt=None):
        H = np.matrix([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            # [0, 0, 1, 0, 0],
        ])
        return H
        # return self.I

    def step(self, measurement, dt=1):
        F = self._get_F(self.X, dt)
        Q = F * F.T * self.Q_multiplier

        self.X = self._f(self.X, dt)
        self.P = F * self.P * F.T + Q

        Z = np.matrix(measurement).T
        Y = Z - self.H * self.X
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.pinv(S)
        self.X = self.X + K * Y
        self.P = (self.I - K * self.H) * self.P

    def get_prediction(self, steps=1, dt=1):
        """Gets prediction or robot position after given number of steps"""
        X = self.X

        for step in range(steps):
            X = self._f(X, dt)

        xy_estimate = X[0, 0], X[1, 0]

        return xy_estimate


class UnscentedKalmanFilter(KalmanFilterBase):
    def __init__(self, *args, **kwargs):
        self.state_size = 6
        self.measurement_size = 2

        super().__init__(*args, **kwargs)

    def step(self, measurement, dt=1):
        # todo tnse
        F = self._get_F(self.X, dt)
        Q = F * F.T * self.Q_multiplier

        self.X = self._f(self.X, dt)
        self.P = F * self.P * F.T + Q

        Z = np.matrix(measurement).T
        Y = Z - self.H * self.X
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.pinv(S)
        self.X = self.X + K * Y
        self.P = (self.I - K * self.H) * self.P
