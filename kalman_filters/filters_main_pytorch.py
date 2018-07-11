from itertools import cycle
import numpy as np

from math import pi
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from torch import optim
from torchviz import make_dot

from kalman_filters.robot import Robot
from kalman_filters.kalman_filter import ExtendedKalmanFilter, KalmanFilterBase, KalmanFilter
from kalman_filters.kalman_filter_pytorch import KalmanFilter as KalmanFiletrPyTorch, ExtendedKalmanFilter as ExtendedKalmanFilterPyTorch
from kalman_filters.utils import distance_between


def run_filter(kalman_filter, target_bot: Robot, starting_point=None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0

    true_positions = []
    measured_positions = []
    filtered_positions = []
    errors = []

    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()

        kalman_filter.step(measurement)
        position_guess = kalman_filter.get_prediction()

        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)

        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize, R: {}, Q: {}, P: {}".format(kalman_filter.R_multiplier, kalman_filter.Q_multiplier, kalman_filter.P_multiplier))

        if starting_point is None or ctr >= starting_point:
            measured_positions.append(measurement)
            filtered_positions.append(position_guess)
            true_positions.append(true_position)
            errors.append(error)

    return measured_positions, filtered_positions, errors, true_positions


def find_optimal(test_target: Robot):
    # kf = KalmanFiletrPyTorch(test_target)
    kf = ExtendedKalmanFilterPyTorch(test_target)

    optimizer = optim.RMSprop(
        [
            {'params': kf.R, 'lr': .5},
            {'params': kf.Q_multiplier, 'lr': .00000001}
        ],
    )

    epochs = 10000
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        kf.print_params()
        optimizer.zero_grad()
        loss = kf.run_filter()
        loss.backward(retain_graph=True)
        optimizer.step()        # fixme тут отваливается
        print(f'Loss: {loss}')
        print()

    # dot_file = make_dot(loss, params=dict(kf.named_parameters()))
    # dot_file.format = 'svg'
    # dot_file.render()

    kf.print_params()
    print(f'Loss: {loss}')


if __name__ == '__main__':
    test_target = Robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)

    measurement_noise = .5 * test_target.distance
    test_target.set_noise(0.0, 0.0, measurement_noise)

    find_optimal(test_target)
    exit()

    # Preparing graph
    figure, axes = plt.subplots(nrows=2, ncols=1)
    figure.set_figheight(10)
    figure.set_figwidth(8)
    ax1, ax2 = axes

    # Run Kalman filter
    R = np.matrix(
        [[140.6324, 130.6976],
         [130.4807, 140.7038]]
    )
    Q = np.matrix(
        [[2.1668, -0.0846, 0.4526, 1.3870, 0.6935],
         [-0.0846, 2.0430, 0.8917, -0.7040, -0.3520],
         [0.4526, 0.8917, 1.0000, 0.0000, 0.0000],
         [1.3870, -0.7040, 0.0000, 2.0000, 1.0000],
         [0.6935, -0.3520, 0.0000, 1.0000, 1.0000]]
    ) * 1e-4
    # kalman_filter = KalmanFilter(Q=Q, R=R)
    kalman_filter = ExtendedKalmanFilter(Q=Q, R=R)
    measured_positions, filtered_positions, errors, true_positions = run_filter(kalman_filter, test_target, starting_point=None)

    # Plot trajectories on top graph
    ax1.scatter(*zip(*measured_positions), s=2, label='Измеренное положение')
    ax1.plot(*zip(*filtered_positions), linewidth=.6, label='Отфильтрованное положение')
    ax1.plot(*zip(*true_positions), linewidth=2, label='Истиное положение')
    ax1.legend()

    # Plot errors on bottom
    ax2.plot(errors)

    # Format output
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax2.yaxis.set_minor_formatter(FormatStrFormatter('%g'))
    ax2.tick_params(which='minor', labelsize=8)
    ax2.grid(which='both')
    # ax2.legend()
    plt.show()
