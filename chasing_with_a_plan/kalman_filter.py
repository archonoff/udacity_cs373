import numpy as np
from math import *


class KalmanFilter:
    def __init__(self, measurement=None):
        if measurement is None:
            x_measurement, y_measurement = (0, 0)
        else:
            x_measurement, y_measurement = measurement

        self.I = np.matrix(np.identity(5))
        self.H = np.matrix([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        self.P = self.I * 1000
        self.X = np.matrix([x_measurement, y_measurement, 0.1, 0.1, 0.1]).T
        self.R = np.matrix(np.identity(2)) * 100

    def unpack_X(self, X):
        x = X[0, 0]
        y = X[1, 0]
        v = X[2, 0]
        b = X[3, 0]
        w = X[4, 0]
        return x, y, v, b, w

    def f(self, X, dt):
        x, y, v, b, w = self.unpack_X(X)
        x, y, b = x + sin(b) * v * dt, y + cos(b) * v * dt, b + w * dt
        return np.matrix((x, y, v, b, w)).T

    def get_F(self, X, dt):
        x, y, v, b, w = self.unpack_X(X)
        F = np.matrix(np.identity(5))

        F[0, 2] = sin(b + w * dt) * dt
        F[0, 3] = cos(b + w * dt) * v * dt
        F[0, 4] = cos(b + w * dt) * v * dt**2

        F[1, 2] = cos(b + w * dt) * dt
        F[1, 3] = -sin(b + w * dt) * v * dt
        F[1, 4] = -sin(b + w * dt) * v * dt**2

        F[3, 4] = dt

        return F

    def step(self, measurement, dt=1):
        F = self.get_F(self.X, dt)
        Q = F * F.T * .01

        # Predict
        self.X = self.f(self.X, dt)
        self.P = F * self.P * F.T + Q

        # Update
        Z = np.matrix(measurement).T
        Y = Z - self.H * self.X
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.pinv(S)

        self.X = self.X + K * Y
        self.P = (self.I - K * self.H) * self.P

    def get_prediction(self, steps=1, dt=1):
        X = self.X

        for step in range(steps):
            X = self.f(X, dt)

        xy_estimate = X[0, 0], X[1, 0]

        return xy_estimate
