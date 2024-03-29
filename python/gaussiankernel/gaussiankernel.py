from matplotlib import pyplot as plt
import math
import numpy as np


def gaussianFilter1d(x, sigma):
    return np.exp(x*x / (-2 * sigma * sigma))


def gaussianFilter2d(xy, sigma):
    return np.exp((xy[0] * xy[0] + xy[1] * xy[1]) / (-2 * sigma * sigma))


if __name__ == '__main__':
    maxValue = 5

    x = np.linspace(-maxValue, maxValue, 1000)

    plt.plot(x, gaussianFilter1d(x, 0.5))
    plt.plot(x, gaussianFilter1d(x, 1))
    plt.plot(x, gaussianFilter1d(x, 2))

    plt.legend(["sigma = 0.5", "sigma = 1", "sigma = 2"])

    plt.savefig("gaussian1d.svg", bbox_inches='tight',pad_inches = 0)

    #Plot Gaussian kernel (smooth and distinct)

    def plotGaussFun(maxValue2d, numPoints, sigma):
        step = (maxValue2d * 2 + 1) / numPoints

        x = np.linspace(-maxValue2d, maxValue2d, numPoints)
        y = np.linspace(-maxValue2d, maxValue2d, numPoints)

        #x = np.arange(-maxValue2d, maxValue2d + step, step)
        #y = np.arange(-maxValue2d, maxValue2d + step, step)
        X, Y = np.meshgrid(x, y)

        XY = np.array([X.flatten(), Y.flatten()]).T

        gauss2 = lambda x: gaussianFilter2d(x, sigma)

        res = np.array(list(map(gauss2, XY)))

        res = res.reshape((numPoints, numPoints))

        plt.imshow(res, cmap="viridis")
        plt.xticks([])
        plt.yticks([])
        # plt.yticks(np.arange( 0, numPoints , tickStep), np.arange(-maxValue2d, maxValue2d + 1, 1))


    maxValue2d = 3
    numPoints = 7 # Points per axis

    plt.figure()
    plotGaussFun(3, 7, 3)
    plt.savefig("gaussian2dDiscrete.svg", bbox_inches='tight',pad_inches = 0)

    plt.figure()
    plotGaussFun(3, 1000, 3)
    plt.savefig("gaussian2dContinuous.svg", bbox_inches='tight',pad_inches = 0)

    #plt.show()



