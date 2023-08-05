from matplotlib import pyplot as plt
import numpy as np


def main():
    Sradii = np.array([5, 8, 10])
    Nradii = np.array([1, 2, 3])

    # Copied from appendix A
    timings = {
        "CNLM": [
            0.5496,
            1.0699,
            1.8595,
            1.2920,
            2.5434,
            4.4259,
            1.9646,
            3.8773,
            6.7478,
        ],
        "CNLM-CUDA": [
            0.0150,
            0.0256,
            0.0440,
            0.0391,
            0.0619,
            0.0993,
            0.0838,
            0.1014,
            0.1481,
        ],
        "SWNLM": [
            5.3056,
            20.1380,
            46.5406,
            12.5936,
            48.0044,
            111.3663,
            19.1606,
            73.1481,
            169.9409,
        ],
        "SWNLM-CUDA": [
            0.0428,
            0.1123,
            0.3241,
            0.0981,
            0.2534,
            0.7775,
            0.1420,
            0.3835,
            1.1823,
        ],
    }

    sizeMult = 20

    for alg in timings:
        plt.figure()

        i = 0
        for Sradius in Sradii:
            comparisonsPerPixel = []
            for Nradius in Nradii:
                numComparisons = (Sradius * 2 + 1) ** 2 * (Nradius * 2 + 1) ** 2
                comparisonsPerPixel.append(numComparisons)

            start = i * len(Sradii)
            end = start + len(Sradii)

            axis = plt.scatter(
                comparisonsPerPixel,
                timings[alg][start:end],
                s=sizeMult * Nradii,
                label=Sradius,
            )
            i += 1

        plt.title(alg)
        plt.xlabel("comparisons per pixel")
        plt.ylabel("execution time (s)")
        legend1 = plt.legend(title="search\nradius", loc="upper left")

        legendSizes = axis.legend_elements("sizes", num=3)

        for i in range(len(legendSizes[1])):
            Nradius = Nradii[i]
            legendSizes[1][i] = legendSizes[1][i].replace(
                str(Nradius * sizeMult), str(Nradius)
            )

        plt.legend(*legendSizes, title="neighborhood\nradius", loc="lower right")
        plt.gca().add_artist(legend1)

        plt.savefig(f"execTimesRadius{alg}.svg")


if __name__ == "__main__":
    main()
