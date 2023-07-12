import cv2 as cv

def main():
    filename = "C:\\Users\\Bob\Documents\\Uni\\BachelorThesis\\NonLocalMeans\\images\\house.tiff" #"../../images/house.tiff"
    searchRadius = 10
    neighborRadius = 3
    padding = searchRadius + neighborRadius

    image = cv.imread(filename)

    padded = cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_REFLECT)

    cv.imshow("original", image)
    cv.imshow("padded", padded)
    cv.waitKey()



if __name__ == "__main__":
    main()