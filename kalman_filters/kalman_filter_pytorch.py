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
        self.P = self._get_P(self.X)

        self.I = Variable(torch.eye(self.state_size, dtype=torch.float), requires_grad=False)
        self.H = self._get_H(self.X)

    def get_position(self):
        """Gets last estimated position"""
        xy_estimate = self.X[0, 0], self.X[1, 0]
        return xy_estimate

    def _get_P(self, X, P_multiplier=100, dt=None):
        I = Variable(torch.eye(self.state_size, dtype=torch.float), requires_grad=False)
        P = I * P_multiplier
        return P

    def print_params(self):
        for n, p in self.named_parameters():
            print('Name: {}\n{}'.format(n, p))
        print('R: {}'.format(self.R))

        F = self._get_F(self.X)
        Q = self._get_Q(F)
        print('Q: {}'.format(Q))

    def forward(self, steps=1000):
        return self.run_filter(steps)

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


class KalmanFilter(KalmanFilterBase):
    def _get_Q(self, F=None):
        Q = torch.mm(F, F.transpose(0, 1)) * self.Q_multiplier
        return Q

    def __init__(self, *args, **kwargs):
        self.state_size = 6
        self.measurement_size = 2

        super().__init__(*args, **kwargs)

        # Parameter for Q
        # self.Q_multiplier = Variable(Tensor([0.01]))
        self.Q_multiplier = nn.Parameter(Tensor([0.0001]))
        self.Q = self._get_Q(self._get_F(self.X, dt=1))

        # Parameters for R
        self.R = nn.Parameter(Tensor([
            [10, 0],
            [0, 10],
        ]))

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

    def run_filter(self, steps=1000, optimize_from_measurement=None):
        measurements, true_positions = self.run_bot(steps)
        predictions = []

        for measurement in measurements:
            self.step(measurement)
            position_guess = self.get_prediction()
            self.test_target.move_in_circle()
            predictions.append(position_guess)

        predictions = torch.stack(predictions)
        if optimize_from_measurement is None:
            loss = self.mse_loss(predictions, true_positions)
        else:
            if optimize_from_measurement >= steps:
                raise RuntimeError('optimize_from_measurement parameter should be less than number of steps')
            loss = self.mse_loss(predictions[optimize_from_measurement:], true_positions[optimize_from_measurement:])

        self.X.detach_()
        self.I.detach_()
        self.H.detach_()
        self.P.detach_()
        self.Q.detach_()

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
    # x = X[0, 0]
    # y = X[1, 0]
    # v = X[2, 0]
    # b = X[3, 0]
    # w = X[4, 0]

    def __init__(self, *args, **kwargs):
        self.state_size = 5
        self.measurement_size = 2

        super().__init__(*args, **kwargs)

        # Parameter for Q
        self.Q_multiplier = nn.Parameter(Tensor([0.0001]))
        self.Q = self._get_Q(self._get_F(self.X, dt=1))

        # Parameters for R
        self.R = nn.Parameter(Tensor([
            [10, 0],
            [0, 10],
        ]))

        self.mse_loss = nn.MSELoss()

    def _f(self, X, dt=1):
        new_X = X.clone()

        # new_X[0, 0] = X[0, 0] + torch.sin(X[3, 0]) * X[2, 0] * dt
        # new_X[1, 0] = X[1, 0] + torch.cos(X[3, 0]) * X[2, 0] * dt
        new_X[3, 0] = X[3, 0] + X[4, 0] * dt

        return new_X

    def _init_X(self, x_measurement, y_measurement):
        X = torch.rand((self.state_size, 1), dtype=torch.float)
        X[0, 0] = x_measurement
        X[1, 0] = y_measurement
        X = Variable(X, requires_grad=False)
        return X

    def _get_Q(self, F=None):
        Q = torch.mm(F, F.transpose(0, 1)) * self.Q_multiplier
        return Q

    def _get_F(self, X, dt=1):
        F = Variable(torch.eye(self.state_size, dtype=torch.float), requires_grad=False)

        # F[0, 2] = torch.sin(X[3, 0] + X[4, 0] * dt) * dt
        # F[0, 3] = torch.cos(X[3, 0] + X[4, 0] * dt) * X[2, 0] * dt
        # F[0, 4] = torch.cos(X[3, 0] + X[4, 0] * dt) * X[2, 0] * dt**2
        #
        # F[1, 2] = torch.cos(X[3, 0] + X[4, 0] * dt) * dt
        # F[1, 3] = -torch.sin(X[3, 0] + X[4, 0] * dt) * X[2, 0] * dt
        # F[1, 4] = -torch.sin(X[3, 0] + X[4, 0] * dt) * X[2, 0] * dt**2
        #
        # F[3, 4] = dt

        return F

    def _get_H(self, X, dt=None):
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            # [0, 0, 1, 0, 0],
        ], dtype=np.float32)
        H = torch.from_numpy(H)
        H = Variable(H, requires_grad=False)
        return H

    def step(self, measurement, dt=1):
        x_measurement, y_measurement = measurement

        F = self._get_F(self.X, dt)
        Q = self._get_Q(F)

        self.X = self._f(self.X, dt)
        self.P = torch.mm(torch.mm(F, self.P), F.transpose(0, 1)) + Q

        Z = Tensor([
            [x_measurement],
            [y_measurement]
        ])
        Z = Variable(Z, requires_grad=False)

        Y = Z - torch.mm(self.H, self.X)
        S = torch.mm(torch.mm(self.H, self.P), self.H.transpose(0, 1)) + self.R
        K = torch.mm(torch.mm(self.P, self.H.transpose(0, 1)), S.inverse())

        self.X = self.X + torch.mm(K, Y)
        I = Variable(torch.eye(self.state_size, dtype=torch.float), requires_grad=False)
        self.P = torch.mm((I - torch.mm(K, self.H)), self.P)

    def run_filter(self, steps=1000, optimize_from_measurement=None):
        measurements, true_positions = self.run_bot(steps)
        predictions = []

        for measurement in measurements:
            self.step(measurement)
            position_guess = self.get_prediction()
            self.test_target.move_in_circle()
            predictions.append(position_guess)

        predictions = torch.stack(predictions)
        if optimize_from_measurement is None:
            loss = self.mse_loss(predictions, true_positions)
        else:
            if optimize_from_measurement >= steps:
                raise RuntimeError('optimize_from_measurement parameter should be less than number of steps')
            loss = self.mse_loss(predictions[optimize_from_measurement:], true_positions[optimize_from_measurement:])

        self.X.detach_()
        self.P.detach_()
        self.H.detach_()

        return loss

    def get_prediction(self, steps=1, dt=1):
        """Gets prediction or robot position after given number of steps"""
        X = self.X.clone()

        for step in range(steps):
            X = self._f(X, dt)

        xy_estimate = X[0:2, 0]

        return xy_estimate
