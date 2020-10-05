import cv2 as cv
import numpy as np

# inputs: path = path of image
class Solution():
    def __init__(self, path):
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        self.img_height, self.img_width = self.img.shape

        # cv.imshow('SE', self.SE)
        # cv.waitKey(0)



    # SE = dimensions of Structure Element an array (kernel)
    def Erosion(self, SE):
        ## if we actually wanted to create the shape of the SE ##
        # SE = np.ones((SE[0], SE[1]), np.uint8)*255
        # SE_height, SE_width = SE.shape

        # for now I just use it's height and width
        SE_height, SE_width = SE[0], SE[1]

        new_img = np.zeros((self.img_height, self.img_width), np.uint8)
        for i in range(self.img_height):
            for j in range(self.img_width):
                # if pixel is white
                if self.img[i][j] == 255:
                    # create check
                    check = True

                    for y in range(SE_height):
                        if not check: break
                        for x in range(SE_width):
                            if not check: break
                            x_coord = (j - SE_width//2) + x
                            y_coord = (i - SE_height//2) + y

                            # if in range + pixel is not white then change check to false
                            if y_coord in range(self.img_height) and x_coord in range(self.img_width) and self.img[y_coord][x_coord] != 255:
                                check = False
                    if check:
                        new_img[i][j] = 255

        return new_img

    # SE = dimensions of Structure Element an array (kernel)
    def Dilation(self, SE):
        ## if we actually wanted to create the shape of the SE ##
        # SE = np.ones((SE[0], SE[1]), np.uint8)*255
        # SE_height, SE_width = SE.shape

        # for now I just use it's height and width
        SE_height, SE_width = SE[0], SE[1]

        # new image
        new_img = self.img.copy()

        for i in range(self.img_height):
            for j in range(self.img_width):
                # if pixel is white
                if self.img[i][j] == 255:
                    # print('point', i, j)
                    for y in range(SE_height):
                        for x in range(SE_width):
                            x_coord = (j - SE_width//2) + x
                            y_coord = (i - SE_height//2) + y

                            # print(y_coord, x_coord)
                            if y_coord in range(self.img_height) and x_coord in range(self.img_width):
                                # print("2", y_coord, x_coord)
                                # # print("x", x_coord, "in range", self.img_width, "and y", y_coord, "in range", self.img_height)
                                new_img[y_coord][x_coord] = 255

        # print('h, w', self.img_height, self.img_width)
        return new_img

    def Opening(self, SE):
        # old_img = self.img.copy()
        self.img = self.Erosion(SE)
        self.img = self.Dilation(SE)

        return self.img

    def Closing(self, SE):
        self.img = self.Dilation(SE)
        self.img = self.Erosion(SE)

        return self.img

    def Boundary(self, SE):
        new_img = self.Erosion(SE)

        return self.img - new_img

# experiment
gun = Solution('images/gun.bmp')
gun_dilation = gun.Dilation([1, 1])
cv.imshow('gun_dilation [1x1]', gun_dilation)
cv.waitKey(0)

gun_dilation2 = gun.Dilation([3, 3])
cv.imshow('gun_dilation [3x3]', gun_dilation2)
cv.waitKey(0)

gun_dilation3 = gun.Dilation([5, 5])
cv.imshow('gun_dilation [5x5]', gun_dilation3)
cv.waitKey(0)

gun_erosion = gun.Erosion([1,1])
cv.imshow('gun_erosion [1x1]', gun_erosion)
cv.waitKey(0)

gun_erosion2 = gun.Erosion([3,3])
cv.imshow('gun_erosion [3x3]', gun_erosion2)
cv.waitKey(0)

gun_boundary = gun.Boundary([3,3])
cv.imshow('gun_boundary [3x3]', gun_boundary)
cv.waitKey(0)

gun = Solution('images/gun.bmp')
gun_opening = gun.Opening([3,3])
cv.imshow('gun_opening [3x3]', gun_opening)
cv.waitKey(0)

gun = Solution('images/gun.bmp')
gun_closing = gun.Closing([3,3])
cv.imshow('gun_closing [3x3]', gun_closing)
cv.waitKey(0)

palm = Solution('images/palm.bmp')
palm_dilation = palm.Dilation([1, 1])
cv.imshow('palm_dilation [1x1]', palm_dilation)
cv.waitKey(0)

palm_dilation2 = palm.Dilation([3, 3])
cv.imshow('palm_dilation [3x3]', palm_dilation2)
cv.waitKey(0)

palm_dilation3 = palm.Dilation([5, 5])
cv.imshow('palm_dilation [5x5]', palm_dilation3)
cv.waitKey(0)

palm_erosion = palm.Erosion([1,1])
cv.imshow('palm_erosion [1x1]', palm_erosion)
cv.waitKey(0)

palm_erosion2 = palm.Erosion([3,3])
cv.imshow('palm_erosion [3x3]', palm_erosion2)
cv.waitKey(0)

palm_boundary = palm.Boundary([3,3])
cv.imshow('palm_boundary [3x3]', palm_boundary)
cv.waitKey(0)

palm = Solution('images/palm.bmp')
palm_opening = palm.Opening([3,3])
cv.imshow('palm_opening [3x3]', palm_opening)
cv.waitKey(0)

palm = Solution('images/palm.bmp')
palm_closing = palm.Closing([3,3])
cv.imshow('palm_closing [3x3]', palm_closing)
cv.waitKey(0)
