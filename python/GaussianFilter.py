import cv2
import numpy as np

def main():
    sigma = 40 / 255.0
    k = 10
    kernelRadius = 3
    kernelDiam = kernelRadius * 2 + 1

    inputImage = cv2.imread("../images/house.tiff")
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    inputImage = inputImage.astype(np.float64) / 255

    noise = inputImage.copy()
    cv2.randn(noise, 0, sigma)

    noisyImage = inputImage + noise

    denoisedImage  = cv2.GaussianBlur(noisyImage, (kernelDiam, kernelDiam), k * sigma)

    diff = inputImage - denoisedImage

    cv2.imshow("input Image", inputImage)
    cv2.imshow("noisy Image", noisyImage)
    cv2.imshow("denoised Image", denoisedImage)
    cv2.imshow("Difference", diff)

    cv2.imwrite("gaussianblur/original.tiff", inputImage)
    cv2.imwrite("gaussianblur/noisy.tiff", inputImage)
    cv2.imwrite("gaussianblur/denoised_sigma=6_k=10.tiff", inputImage)
    cv2.imwrite("gaussianblur/difference.tiff", inputImage)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
