import cv2 as cv
import os


def main():
    numResizes = 5
    extension = ".png"
    outDir = "./out/"
    inDir = "./images/"

    os.makedirs(outDir, exist_ok =True)

    for file in os.listdir(inDir):
        filename = file.partition(".")[0]
        photo = cv.imread(inDir + file)
        photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

        for i in range(numResizes):
            rows, cols = photo.shape

            newFilename = outDir + filename + "_" + str(cols) + "x" + str(rows) + extension
            cv.imshow(newFilename, photo)
            cv.imwrite(newFilename, photo)

            newSize = (cols // 2, rows // 2,)
            photo = cv.resize(photo, newSize, interpolation = cv.INTER_AREA)

    cv.waitKey()


if __name__ == "__main__":
    main()
