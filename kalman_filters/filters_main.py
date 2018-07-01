from math import pi
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from kalman_filters.robot import Robot
from kalman_filters.kalman_filter import ExtendedKalmanFilter, KalmanFilterBase
from kalman_filters.utils import distance_between


def run_filter(kalman_filter: KalmanFilterBase, target_bot: Robot):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0

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
            print("You got it right! It took you ", ctr, " steps to localize.")
            localized = True
        if ctr == 1000:
            print("Sorry, it took you too many steps to localize the target.")

        measured_positions.append(measurement)
        filtered_positions.append(position_guess)

    figure, axes = plt.subplots(nrows=2, ncols=1)   #, sharex=True, gridspec_kw={'height_ratios': (5, 1)})
    figure.set_figheight(10)
    figure.set_figwidth(8)

    ax1, ax2 = axes

    ax1.scatter(*zip(*measured_positions), s=2, label='Измеренное положение')
    ax1.plot(*zip(*filtered_positions), linewidth=.2, label='Отфильтрованное положение')
    ax1.legend()

    ax2.plot(errors)
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax2.yaxis.set_minor_formatter(FormatStrFormatter('%g'))
    ax2.tick_params(which='minor', labelsize=8)
    ax2.grid(which='both')

    plt.show()

    return localized


if __name__ == '__main__':
    # test_target = Robot(2.1, 4.3, 0.5, 2 * pi / 34.0, 1.5)        # From "Adding noise" exercise
    test_target = Robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)             # From "The final hunt" exercise

    measurement_noise = .05 * test_target.distance
    test_target.set_noise(0.0, 0.0, measurement_noise)

    mode = 'ekf'

    if mode == 'ekf':
        kalman_filter = ExtendedKalmanFilter(R_multiplier=100, Q_multiplier=.0001, P_multiplier=100000)

    run_filter(kalman_filter, test_target)
