import cv2 as cv
import matplotlib.pyplot as plt

def main():
    noisyGaussFile = "Gblur_NoisyGauss.png"
    noisyGauss = cv.imread(noisyGaussFile)
    multiplier = 5

    noisy = cv.imread("Gblur_Ni.png")
    plt_noisy = cv.cvtColor(noisy, cv.COLOR_BGR2RGB)
    noisyplot = plt.imshow(plt_noisy)

    plt.figure()
    plt_noisyGauss = cv.cvtColor(noisyGauss, cv.COLOR_BGR2RGB)
    imgplot = plt.imshow(plt_noisyGauss * multiplier)
    cv.imwrite("Gblur_NoisyGaussMult.png", noisyGauss * multiplier)

    x = sum(sum(noisyGauss))
    print(x)


    plt.figure()
    cnlmNiNj2G = cv.imread("CNLM_Ni-Nj2G.png")

    m2 = 15

    plt.imshow(cnlmNiNj2G * m2)
    cv.imwrite("CNLM_Ni-Nj2GMult.png", cnlmNiNj2G * m2)


    CNLMnoisyWithRect = cv.imread("CNLM_noisyWithRect.png")
    cropped = CNLMnoisyWithRect[0:128, 0:128]
    plt.figure()
    plt.imshow(cropped)

    cv.imwrite("CNLM_noisyWithRectCropped.png", cropped)


    plt.show()

if __name__ == "__main__":
    main()

