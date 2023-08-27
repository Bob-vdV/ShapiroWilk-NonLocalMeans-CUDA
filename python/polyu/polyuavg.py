import numpy as np
import matplotlib.pyplot as plt


def main():
    filename = "avgresultspolyu.csv"

    data = np.genfromtxt(filename, delimiter=",", dtype=str)

    data = data[1:]

    algorithms = np.unique(data[:, 1])

    print(data)
    print()

    plt.figure()
    ax = plt.axes()

    plt.yscale("log")

    plt.xlabel("image resolution")
    plt.ylabel("execution time (s)")

    for alg in algorithms:
        algData = data[np.where(data[:, 1] == alg)]

        execTimes = algData[:, 2].astype(np.double)
        resolutions = algData[:, 0]

        ax.plot(resolutions, execTimes, label=alg, marker="o")

    handles, labels = ax.get_legend_handles_labels()

    handles = [handles[2], handles[0], handles[3], handles[1]]
    labels = [labels[2], labels[0], labels[3], labels[1]]

    ax.legend(handles, labels, title="algorithm")

    plt.savefig(f"polyUExecTimeAvg.svg", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
