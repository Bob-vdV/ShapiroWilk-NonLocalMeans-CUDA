import numpy as np
import matplotlib.pyplot as plt


def main():
    filename = "C:/Users/Bob/Documents/Uni/BachelorThesis/NonLocalMeans/python/polyu/polyUExecTimes.csv"

    data = np.genfromtxt(filename, delimiter=",", dtype=str)

    data = data[1:]

    images = np.unique(data[:, 0])
    algorithms = np.unique(data[:, 2])

    print(data)
    print()

    for image in images:
        imageData = data[np.where(data[:, 0] == image)]

        plt.figure()
        plt.yscale("log")
        plt.title(image)

        plt.xlabel("image resolution")
        plt.ylabel("execution time (s)")

        for alg in algorithms:
            algData = imageData[np.where(imageData[:, 2] == alg)]

            execTimes = algData[:, 3].astype(np.double)
            resolutions = algData[:, 1]

            plt.plot(resolutions, execTimes, label=alg, marker="o")

        plt.legend(title="algorithm")

        plt.savefig(f"polyUExecTime{image}.svg", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
