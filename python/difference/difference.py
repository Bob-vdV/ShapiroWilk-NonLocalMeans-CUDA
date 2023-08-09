import cv2 as cv


def main():
    algs = ["cnlm", "swnlm"]
    images = ["boat", "house", "mandril"]

    cuda = "cuda"
    png = ".png"

    for image in images:
        for alg in algs:
            seqVersion = cv.imread(image + alg + png)
            cudaVersion = cv.imread(image + alg + cuda + png)

            diff = cv.absdiff(seqVersion, cudaVersion)
            print(f"{image} {alg}: {diff.max()}")

            cv.imwrite(image + alg + "diff" + png, diff)

            cv.imwrite(image + alg + "absdiff" + png, diff * 255)


if __name__ == "__main__":
    main()
