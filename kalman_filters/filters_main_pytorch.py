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
from kalman_filters.kalman_filter_pytorch import KalmanFilter as KalmanFiletrPyTorch
from kalman_filters.utils import distance_between


def run_filter(kalman_filter, target_bot: Robot):
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
        errors.append(error)

        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize, R: {}, Q: {}, P: {}".format(kalman_filter.R_multiplier, kalman_filter.Q_multiplier, kalman_filter.P_multiplier))
            # localized = True
        # if ctr == 1000:
        #     print("Sorry, it took you too many steps to localize the target.")

        measured_positions.append(measurement)
        filtered_positions.append(position_guess)
        true_positions.append(true_position)

    return measured_positions, filtered_positions, errors, true_positions


def find_optimal(test_target: Robot):
    kf = KalmanFiletrPyTorch(test_target)
    optimizer = optim.Adam(
        [
            {'params': kf.R, 'lr': .1},
            {'params': kf.Q_multiplier, 'lr': .01}
        ],
    )

    epochs = 10000
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        kf.print_params()
        optimizer.zero_grad()
        loss = kf.run_filter()
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss}')
        print()

    # dot_file = make_dot(loss, params=dict(kf.named_parameters()))
    # dot_file.format = 'svg'
    # dot_file.render()

    kf.print_params()
    print(f'Loss: {loss}')

    # Last iteration results
    '''
    Loss: 5.516384124755859
    Name: R
    Parameter
    containing:
    tensor([[0.1026, 6.8335],
            [6.0354, -0.1011]])
    Name: Q
    Parameter
    containing:
    tensor(1.00000e-03 *
           [[0.0286, 0.0627, 0.0196, 0.0517, 0.0362, 0.0394],
            [0.0504, 0.0549, 0.0341, 0.0011, 0.0410, 0.0748],
            [0.0143, 0.0955, 0.0090, 0.0505, 0.0527, 0.1450],
            [0.0228, 0.0302, 0.1236, 0.0619, 0.4331, -0.0110],
            [0.0801, 0.0031, 0.0550, -0.0268, -2.2315, 0.8949],
            [0.0161, -0.0073, -0.2275, 0.1143, 1.5691, 0.2306]])
    '''

    # Results using Q multiplicator
    '''
    Name: R
    Parameter containing:
    tensor([[ 8.4598,  7.9919],
            [ 3.9773,  6.2945]])
    Name: Q_multiplier
    Parameter containing:
    tensor(1.00000e-02 *
           [ 2.5113])
    Q: tensor(1.00000e-02 *
           [[ 7.5243,  0.0000,  5.0162,  0.0000,  2.5081,  0.0000],
            [ 0.0000,  7.5243,  0.0000,  5.0162,  0.0000,  2.5081],
            [ 5.0162,  0.0000,  5.0162,  0.0000,  2.5081,  0.0000],
            [ 0.0000,  5.0162,  0.0000,  5.0162,  0.0000,  2.5081],
            [ 2.5081,  0.0000,  2.5081,  0.0000,  2.5081,  0.0000],
            [ 0.0000,  2.5081,  0.0000,  2.5081,  0.0000,  2.5081]])
    Loss: 3.061612367630005
    '''

    # After detaching
    '''
    Name: R
    Parameter containing:
    tensor([[ 4.8707,  6.4799],
            [ 3.5715,  8.0508]])
    Name: Q_multiplier
    Parameter containing:
    tensor(1.00000e-02 *
           [ 2.5243])
    Q: tensor(1.00000e-02 *
           [[ 7.5637,  0.0000,  5.0425,  0.0000,  2.5212,  0.0000],
            [ 0.0000,  7.5637,  0.0000,  5.0425,  0.0000,  2.5212],
            [ 5.0425,  0.0000,  5.0425,  0.0000,  2.5212,  0.0000],
            [ 0.0000,  5.0425,  0.0000,  5.0425,  0.0000,  2.5212],
            [ 2.5212,  0.0000,  2.5212,  0.0000,  2.5212,  0.0000],
            [ 0.0000,  2.5212,  0.0000,  2.5212,  0.0000,  2.5212]])
    Loss: 3.0173444747924805
    '''

    # Sharp transition
    '''
    Epoch: 17
    Name: R
    Parameter containing:
    tensor([[ 8.5739,  5.6576],
            [ 9.8287,  6.3844]])
    Name: Q_multiplier
    Parameter containing:
    tensor(1.00000e-02 *
           [ 3.3377])
    Q: tensor([[ 0.1000,  0.0000,  0.0667,  0.0000,  0.0333,  0.0000],
            [ 0.0000,  0.1000,  0.0000,  0.0667,  0.0000,  0.0333],
            [ 0.0667,  0.0000,  0.0667,  0.0000,  0.0333,  0.0000],
            [ 0.0000,  0.0667,  0.0000,  0.0667,  0.0000,  0.0333],
            [ 0.0333,  0.0000,  0.0333,  0.0000,  0.0333,  0.0000],
            [ 0.0000,  0.0333,  0.0000,  0.0333,  0.0000,  0.0333]])
    Loss: 1655.6890869140625

    Epoch: 18
    Name: R
    Parameter containing:
    tensor([[ 6.6679e+08, -1.0173e+09],
            [-5.8558e+08,  8.9341e+08]])
    Name: Q_multiplier
    Parameter containing:
    tensor([ 26994.2305])
    Q: tensor([[ 0.1001,  0.0000,  0.0668,  0.0000,  0.0334,  0.0000],
            [ 0.0000,  0.1001,  0.0000,  0.0668,  0.0000,  0.0334],
            [ 0.0668,  0.0000,  0.0668,  0.0000,  0.0334,  0.0000],
            [ 0.0000,  0.0668,  0.0000,  0.0668,  0.0000,  0.0334],
            [ 0.0334,  0.0000,  0.0334,  0.0000,  0.0334,  0.0000],
            [ 0.0000,  0.0334,  0.0000,  0.0334,  0.0000,  0.0334]])
    Loss: 20.964256286621094
    '''

    # Adam stopped optimizing
    '''
    Epoch: 208
    Name: R
    Parameter containing:
    tensor([[ 7.7792,  7.1198],
            [ 6.5951,  8.5665]])
    Name: Q_multiplier
    Parameter containing:
    tensor([ 0.1786])
    Q: tensor([[ 0.5358,  0.0000,  0.3572,  0.0000,  0.1786,  0.0000],
            [ 0.0000,  0.5358,  0.0000,  0.3572,  0.0000,  0.1786],
            [ 0.3572,  0.0000,  0.3572,  0.0000,  0.1786,  0.0000],
            [ 0.0000,  0.3572,  0.0000,  0.3572,  0.0000,  0.1786],
            [ 0.1786,  0.0000,  0.1786,  0.0000,  0.1786,  0.0000],
            [ 0.0000,  0.1786,  0.0000,  0.1786,  0.0000,  0.1786]])
    Loss: 2.176464319229126
    '''


if __name__ == '__main__':
    test_target = Robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)

    measurement_noise = .05 * test_target.distance
    test_target.set_noise(0.0, 0.0, measurement_noise)

    # find_optimal(test_target)
    # exit()

    # Preparing graph
    figure, axes = plt.subplots(nrows=2, ncols=1)
    figure.set_figheight(10)
    figure.set_figwidth(8)
    ax1, ax2 = axes

    # Run Kalman filter
    R = np.matrix(
        [[7.7792, 7.1198],
         [6.5951, 8.5665]]
    )
    Q = np.matrix(
        [[0.5358, 0.0000, 0.3572, 0.0000, 0.1786, 0.0000],
         [0.0000, 0.5358, 0.0000, 0.3572, 0.0000, 0.1786],
         [0.3572, 0.0000, 0.3572, 0.0000, 0.1786, 0.0000],
         [0.0000, 0.3572, 0.0000, 0.3572, 0.0000, 0.1786],
         [0.1786, 0.0000, 0.1786, 0.0000, 0.1786, 0.0000],
         [0.0000, 0.1786, 0.0000, 0.1786, 0.0000, 0.1786]]
    )# * 1.00000e-02
    kalman_filter = KalmanFilter(Q=Q, R=R)
    measured_positions, filtered_positions, errors, true_positions = run_filter(kalman_filter, test_target)

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
