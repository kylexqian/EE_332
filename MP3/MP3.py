import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Solution():
    def __init__(self, path):
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        self.img_height, self.img_width = self.img.shape

        self.H = None
        self.T = None

    def HistoEqualization(self):
        # create histogram, 0 for each bin at start. Range is 255 since that's the range of grayscale pixels
        H = [0 for _ in range(256)]

        # fill histogram
        for i in range(self.img_height):
            for j in range(self.img_width):
                H[self.img[i][j]] += 1

        # graph histogram (flatten self.img to get 1D array)
        Histogram = plt.figure()
        plt.hist(self.img.flatten(), [i for i in range(256)])
        plt.xlabel('Gray scale color')
        plt.ylabel('# of pixels')
        Histogram.suptitle('Histogram of pixel colors')
        Histogram.savefig('Results/Moon Histogram.png')

        # transformation function
        T = [0 for _ in range(256)]

        # first value is same
        T[0] = H[0]

        # accumulate values for the rest
        for i in range(1,256):
            T[i] = T[i-1]
            T[i] += H[i]

        # normalize
        for i in range(256):
            T[i] = T[i]/T[-1]

        # graph T
        Transformation = plt.figure()
        plt.plot([i for i in range(256)], T)
        plt.xlabel('Gray scale color')
        plt.ylabel('Normalized Integral Accumulation')
        Transformation.suptitle('Transformation Function')
        Transformation.savefig('Results/Moon Transformation.png')

        self.T = T
        self.H = H

        # use transformation to recolor image
        new_img = self.img.copy()
        for i in range(self.img_height):
            for j in range(self.img_width):
                color = self.img[i][j]
                new_img[i][j] = int(T[color] * 255)

        self.img = new_img
        return new_img

    def Linear_Lighting(self):
        # create plane
        plane = np.copy(self.img)

        # obtain x in x = A^t * y, where we psuedo inverse A and dot product that with y
        # first build y
        y = []
        for i in range(self.img_height):
            for j in range(self.img_width):
                y.append(plane[i][j])
        y = np.array(y)

        # now build A^t
        A = []
        for i in range(self.img_height):
            for j in range(self.img_width):
                A.append([i, j, 1])

        # now perform linear algebra to get x
        x = np.dot(np.linalg.pinv(A), y)

        # fit plane with x
        for i in range(self.img_height):
            for j in range(self.img_width):
                num = x[0]*i + x[1]*j +x[2]
                if num > 255:
                    plane[i][j] = 255
                elif num < 0:
                    plane[i][j] = 0
                else:
                    plane[i][j] = num

        # print/show linear lighting
        cv.imshow('linear', plane)
        cv.imwrite('Results/Linear Lighting.bmp', plane)
        cv.waitKey(0)

        # lighting correction using truncated
        # first get min and max to find avg
        min_num = 0
        max_num = 0
        for i in range(self.img_height):
            for j in range(self.img_width):
                min_num = min((int(self.img[i][j]) - int(plane[i][j])), min_num)
                max_num = max((int(self.img[i][j]) - int(plane[i][j])), max_num)

        # find average of max and min
        avg = (max_num + min_num)/2

        # using truncated version, so if color > 255 just set to 255 and if color < 0 just set to 0
        truncated = self.img.copy()
        for i in range(self.img_height):
            for j in range(self.img_width):
                new_color = int(self.img[i][j]) - int(plane[i][j]) + avg + 128
                if new_color < 0:
                    truncated[i][j] = 0
                elif new_color > 255:
                    truncated[i][j] = 255
                else:
                    truncated[i][j] = new_color

        return truncated

    def Quadratic_Lighting(self):
        # create plane
        plane = np.copy(self.img)

        # obtain x in x = A^t * y, where we psuedo inverse A and dot product that with y
        # first build y
        y = []
        for i in range(self.img_height):
            for j in range(self.img_width):
                y.append(plane[i][j])
        y = np.array(y)

        # now build A^t
        A = []
        for i in range(self.img_height):
            for j in range(self.img_width):
                A.append([i**2, i*j, j**2, i, j, 1])

        # now perform linear algebra to get x
        x = np.dot(np.linalg.pinv(A), y)

        # fit plane with x
        for i in range(self.img_height):
            for j in range(self.img_width):
                num = x[0]*(i**2) + x[1]*i*j + x[2]*(j**2) + x[3]*i + x[4]*j + x[5]
                if num > 255:
                    plane[i][j] = 255
                elif num < 0:
                    plane[i][j] = 0
                else:
                    plane[i][j] = num

        # print/show quadratic lighting
        cv.imshow('quadratic', plane)
        cv.imwrite('Results/Quadratic Lighting.bmp', plane)
        cv.waitKey(0)

        # lighting correction using truncated
        # first get min and max to find avg
        min_num = 0
        max_num = 0
        for i in range(self.img_height):
            for j in range(self.img_width):
                min_num = min((int(self.img[i][j]) - int(plane[i][j])), min_num)
                max_num = max((int(self.img[i][j]) - int(plane[i][j])), max_num)

        # find average of max and min
        avg = (max_num + min_num)/2

        # using truncated version, so if color > 255 just set to 255 and if color < 0 just set to 0
        truncated = self.img.copy()
        for i in range(self.img_height):
            for j in range(self.img_width):
                new_color = int(self.img[i][j]) - int(plane[i][j]) + avg + 128
                if new_color < 0:
                    truncated[i][j] = 0
                elif new_color > 255:
                    truncated[i][j] = 255
                else:
                    truncated[i][j] = new_color

        return truncated

moon = Solution("images/moon.bmp")
recolor = moon.HistoEqualization()
linear_light = moon.Linear_Lighting()
# cv.imshow('moon recolored', recolor)
# cv.imwrite('Results/Moon Recoloring.bmp', recolor)
# cv.waitKey(0)
#
# cv.imshow('moon linear lighted', linear_light)
# cv.imwrite('Results/Moon Linear Lighting.bmp', linear_light)
# cv.waitKey(0)

# reset
moon.img = recolor
quad_light = moon.Quadratic_Lighting()
# cv.imshow('moon quad lighted', quad_light)
# cv.imwrite('Results/Moon Quadratic Lighting.bmp', quad_light)
# cv.waitKey(0)
