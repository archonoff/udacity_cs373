from itertools import cycle

from math import pi
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
    optimizer = optim.SGD(
        [
            {'params': kf.R, 'lr': .1},
            {'params': kf.Q, 'lr': .000000001}
        ],
        # lr=.01,
    )

    epochs = 100
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        kf.print_params()
        optimizer.zero_grad()
        loss = kf.run_filter()
        # make_dot(loss, params=dict(kf.named_parameters()))
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f'Loss: {loss}')

    for n, p in kf.named_parameters():
        print('Name: {}\n{}'.format(n, p))

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


if __name__ == '__main__':
    test_target = Robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)

    measurement_noise = .05 * test_target.distance
    test_target.set_noise(0.0, 0.0, measurement_noise)

    find_optimal(test_target)

    exit()

    # Preparing graph
    figure, axes = plt.subplots(nrows=2, ncols=1)
    figure.set_figheight(10)
    figure.set_figwidth(8)
    ax1, ax2 = axes

    # Run Kalman filter
    kalman_filter = KalmanFiletrPyTorch(test_target)
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
    ax2.legend()
    plt.show()
