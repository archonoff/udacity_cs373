import numpy as np
from math import *
import torch
from torch.autograd import Variable
from torch import Tensor
from torch import nn

from kalman_filters.robot import Robot


class KalmanFilterBase(nn.Module):
    state_size = NotImplemented
    measurement_size = NotImplemented

    def __init__(self, test_target: Robot, measurement=None):
        super().__init__()      # PyTorch __init__

        if measurement is None:
            x_measurement, y_measurement = (0, 0)
        else:
            x_measurement, y_measurement = measurement

        self.test_target = test_target

        self.X = self._init_X(x_measurement, y_measurement)
        self.I = Variable(torch.eye(self.state_size, dtype=torch.float), requires_grad=False)
        self.H = self._get_H(self.X)
        self.P = self._get_P(self.X)
        self.R = self._get_R(self.X)

    def get_position(self):
        """Gets last estimated position"""
        xy_estimate = self.X[0, 0], self.X[1, 0]
        return xy_estimate

    def _get_P(self, X, P_multiplier=100, dt=None):
        # P = 10 * torch.rand((self.state_size, self.state_size), dtype=torch.float)
        P = self.I * P_multiplier
        return P

    def _get_R(self, X, dt=None):
        R = 10 * torch.rand((self.measurement_size, self.measurement_size), dtype=torch.float)
        R = nn.Parameter(R)
        return R

    def print_params(self):
        for n, p in self.named_parameters():
            print('Name: {}\n{}'.format(n, p))
        print('Q: {}'.format(self.Q))


class KalmanFilter(KalmanFilterBase):
    def _get_Q(self, F=None):
        Q = torch.mm(F, F.transpose(0, 1)) * self.Q_multiplier
        return Q

    def __init__(self, *args, **kwargs):
        self.state_size = 6
        self.measurement_size = 2

        super().__init__(*args, **kwargs)

        self.Q_multiplier = nn.Parameter(Tensor([0.0001]))
        self.Q = self._get_Q(self._get_F(self.X, dt=1))
        self.mse_loss = nn.MSELoss()

    def _init_X(self, x_measurement, y_measurement):
        X = torch.rand((self.state_size, 1), dtype=torch.float)
        X[0, 0] = x_measurement
        X[1, 0] = y_measurement
        X = Variable(X, requires_grad=False)
        return X

    def _get_F(self, X, dt=None):
        F = np.array([
            [1.0, 0, dt, 0, dt**2, 0],
            [0, 1, 0, dt, 0, dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        F = torch.from_numpy(F)
        F = Variable(F, requires_grad=False)
        return F

    def _get_H(self, X, dt=None):
        H = np.array([[1.0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        H = torch.from_numpy(H)
        H = Variable(H, requires_grad=False)
        return H

    def step(self, measurement, dt=1):
        x_measurement, y_measurement = measurement

        F = self._get_F(self.X, dt)
        self.Q = self._get_Q(F)

        self.X = torch.mm(F, self.X)
        self.P = torch.mm(torch.mm(F, self.P), F.transpose(0, 1)) + self.Q

        Z = Tensor([
            [x_measurement],
            [y_measurement]
        ])
        Z = Variable(Z, requires_grad=False)

        Y = Z - torch.mm(self.H, self.X)
        S = torch.mm(torch.mm(self.H, self.P), self.H.transpose(0, 1)) + self.R
        K = torch.mm(torch.mm(self.P, self.H.transpose(0, 1)), S.inverse())
        self.X = self.X + torch.mm(K, Y)
        self.P = torch.mm((self.I - torch.mm(K, self.H)), self.P)

    def run_bot(self, steps=1000):
        measurements = []
        true_positions = []
        for step in range(steps):
            measurement = self.test_target.sense()
            true_position = (self.test_target.x, self.test_target.y)

            measurements.append(measurement)
            true_positions.append(true_position)

            self.test_target.move_in_circle()

        return measurements, Variable(Tensor(true_positions), requires_grad=False)

    def forward(self, steps=1000):
        return self.run_filter(steps)

    def run_filter(self, steps=1000):
        measurements, true_positions = self.run_bot(steps)
        predictions = []

        for measurement in measurements:
            self.step(measurement)
            position_guess = self.get_prediction()
            self.test_target.move_in_circle()
            predictions.append(position_guess)

        predictions = torch.stack(predictions)
        loss = self.mse_loss(predictions, true_positions)

        return loss

    def get_prediction(self, steps=1, dt=1):
        """Gets prediction or robot position after given number of steps"""
        X = self.X

        for step in range(steps):
            F = self._get_F(X, dt)
            X = torch.mm(F, X)

        xy_estimate = X[0:2, 0]

        return xy_estimate


class ExtendedKalmanFilter(KalmanFilterBase):
    prev_x = None
    prev_y = None
    prev_b = None

    def __init__(self, *args, **kwargs):
        self.state_size = 5
        self.measurement_size = 3

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
            [0, 0, 1, 0, 0],
        ])
        return H
        # return self.I

    def step(self, measurement, dt=1):
        # Extract more data from measurements
        current_x, current_y = measurement

        if self.prev_x is not None and self.prev_y is not None:
            dx = current_x - self.prev_x
            dy = current_y - self.prev_y
            current_dist = sqrt(dx**2 + dy**2)
            v = current_dist / dt
            b = atan2(dy, dx)
        else:
            v = 0
            b = 0

        if self.prev_b is not None:
            db = b - self.prev_b
            w = db / dt
        else:
            w = 0

        self.prev_x = current_x
        self.prev_y = current_y
        self.prev_b = b

        # measurement: measured x, measured y, linear velocity, current beta, angular velocity
        # measurement = current_x, current_y, v, b, w
        measurement = current_x, current_y, v

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
