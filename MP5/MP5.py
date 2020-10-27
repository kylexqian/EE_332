import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys

class Solution():
    def __init__(self, path, title):
        # image
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if self.img is None:
            raise NameError('image not found')

        self.title = title

    # generate gaussian kernel
    def Generate_GKernel(self, N, Sigma):
        # get dimensions
        height = N[0]
        width = N[1]
        if not height%2 or not width%2:
            raise NameError('Kernel Size must be odd number')
        center = [height//2, width//2]

        # create kernel
        kernel = np.zeros((height,width))

        # fill kernel
        for i in range(height):
            for j in range(width):
                # formula for 2D G(x) = exp(-(x^2+y^2)/(2*sigma^2))/(2*pi*sigma^2), where x is distance
                x = i-center[0]
                y = j-center[0]
                kernel[i][j] = np.exp(-(x**2+y**2)/(2*(Sigma**2)))/(2*np.pi*(Sigma**2))

        # normalize
        total = np.sum(kernel)
        kernel = kernel/total

        return kernel

    # use gaussian kernel to smooth image
    def GaussSmoothing(self, N, Sigma):
        # Create gaussian kernel
        kernel = self.Generate_GKernel(N, Sigma)

        smooth_img = signal.convolve2d(self.img, kernel)
        smooth_img = smooth_img.astype(np.uint8)

        # print(smooth_img)
        # cv.imshow('smoothed ' + str(N) + ', sigma: ' + str(Sigma) + ' ' + self.title, np.absolute(smooth_img))
        # cv.imwrite('smoothed ' + str(N) + ', sigma: ' + str(Sigma) + ' ' + self.title + '.jpg', np.absolute(smooth_img))
        # cv.waitKey(0)

        return smooth_img


    # generate image gradient, creating magnitude and direction. 3 modes: Robert, Sobel, Prewitt
    def ImageGradient(self, img, mode='Sobel'):
        if mode == 'Robert':
            x_kernel = np.array([[1,0],
                                [0,-1]])
            y_kernel = np.array([[0,-1j],
                                [1j,0]])
            kernel = x_kernel + y_kernel
        elif mode == 'Sobel':
            x_kernel = np.array([[-1,0,1],
                                [-2,0,2],
                                [-1,0,1]])
            y_kernel = np.array([[1j,2j,1j],
                                [0,0,0],
                                [-1j,-2j,-1j]])
            kernel = x_kernel + y_kernel
        elif mode == 'Prewitt':
            x_kernel = np.array([[-1,0,1],
                                [-1,0,1],
                                [-1,0,1]])
            y_kernel = np.array([[1j,1j,1j],
                                [0,0,0],
                                [-1j,-1j,-1j]])
            kernel = x_kernel + y_kernel

        # create magnitude and direction images
        magnitude = np.zeros(img.shape)
        direction = np.zeros(img.shape)

        # convolute kernel with image -> note this assumes x_kernel and y_kernel are the same shape
        gradient = signal.convolve2d(img, kernel, )

        # magnitude
        magnitude = np.absolute(gradient)
        # magnitude = magnitude/np.max(magnitude)

        # direction in degrees
        direction = np.degrees(np.angle(gradient))

        # showing image
        magnitude_img = magnitude/np.max(magnitude) * 255
        magnitude_img = magnitude_img.astype(np.uint8)
        # direction_img = direction + np.abs(np.min(direction))
        # direction_img = direction_img/np.max(direction)*255
        # direction_img = direction_img.astype(np.uint8)

        # cv.imshow('magnitude ' + mode + ' ' + self.title, magnitude_img)
        # cv.imwrite('magnitude ' + mode + ' ' + self.title + '.jpg', magnitude_img)
        # cv.waitKey(0)
        # cv.imshow('angle ' + mode + ' ' + self.title, direction_img)
        # cv.waitKey(0)

        return magnitude, direction

    def FindThreshold(self, magnitude, percentageOfNonEdge=0.95):
        self.pne = str(percentageOfNonEdge)
        T_high = 0
        magnitude = magnitude.astype(np.uint8)
        height, width = magnitude.shape
        histogram = np.zeros(256)
        max_n = 0
        h_sum = 0

        # max magnitude
        max_n = np.max(magnitude)

        # build histogram
        for i in range(height):
            for j in range(width):
                histogram[magnitude[i][j]] += 1

        # normalize
        histogram = histogram/np.sum(histogram)

        # find threshold
        for i in range(max_n):
            if h_sum >= percentageOfNonEdge:
                T_high = i
                break
            h_sum += histogram[i]

        return T_high, T_high/2

    def NonmaximaSupress(self, magnitude, direction):
        height, width = magnitude.shape
        supressed = np.zeros(magnitude.shape)
        direction[direction < 0] += 180


        for i in range(1,height-1):
            for j in range(1,width-1):
                # 1 & 5
                if (0 <= direction[i][j] < 22.5) or (157.5 <= direction[i][j] <= 180):
                    p = [0, 1]
                # 2 & 6
                elif (22.5 <= direction[i][j] < 67.5):
                    p = [-1,1]
                # 3 & 7
                elif (67.5 <= direction[i][j] < 112.5):
                    p = [-1,0]
                # 4 & 8
                elif (112.5 <= direction[i][j] < 157.5):
                    p = [-1,-1]

                # if local max -> fill in with magnitude
                if (magnitude[i][j] >= magnitude[i+p[0]][j+p[1]]) and (magnitude[i][j] >= magnitude[i-p[0]][j-p[1]]):
                    supressed[i,j] = magnitude[i][j]

        # for viewing
        supressed_img = supressed.astype(np.uint8)

        # cv.imshow('Supressed Nonmaxima ' + self.title, supressed_img)
        # cv.imwrite('Supressed Nonmaxima ' + self.title + '.jpg', supressed_img)
        # cv.waitKey(0)

        return supressed

    def EdgeLinking(self, supressed, T_high, T_low):
        # generate mag_high and mag_low
        mag_high = np.zeros(supressed.shape)
        mag_low = np.zeros(supressed.shape)
        height, width = supressed.shape

        for i in range(height):
            for j in range(width):
                if supressed[i][j] > T_high:
                    mag_high[i][j] = supressed[i][j]
                if supressed[i][j] > T_low:
                    mag_low[i][j] = supressed[i][j]

        # for viewing
        mh = mag_high.astype(np.uint8)
        ml = mag_low.astype(np.uint8)

        cv.imshow('mhigh ' + self.pne + ' ' + self.title, mh)
        cv.imwrite('mhigh ' + self.pne + ' ' + self.title + '.jpg', mh)
        cv.waitKey(0)
        cv.imshow('mlow ' + self.pne + ' ' + self.title, ml)
        cv.imwrite('mlow ' + self.pne + ' ' + self.title + '.jpg', ml)
        cv.waitKey(0)

        # recursively generate edge_link
        edge_link = np.copy(mag_high)
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1)]

        # recurse through mag_low, if it has a mag_high link at an adjacent pixel -> add it to edge_link
        sys.setrecursionlimit(10**5)

        def helper(i, j):
            for direction in directions:
                y = i + direction[0]
                x = j + direction[1]
                if y in range(height) and x in range(width) and \
                mag_low[y][x] > 0 and edge_link[y][x] == 0:
                    edge_link[i][j] = mag_low[i][j]
                    helper(y,x)

        for i in range(height):
            for j in range(width):
                if edge_link[i][j] == 0 and mag_low[i][j] > 0:
                    helper(i, j)

        edge_link = edge_link.astype(np.uint8)
        # cv.imshow('Linked image ' + self.title, edge_link)
        # cv.imwrite('Linked image ' + self.title + '.jpg', edge_link)
        # cv.waitKey(0)

        print('sanity check', np.array_equal(edge_link, mag_low), np.array_equal(edge_link, mag_high))

        return edge_link

### Experiment ###
# lena = Solution('images/lena.bmp', 'lena')
# smooth_img = lena.GaussSmoothing([3,3], 1)
# magnitude, direction = lena.ImageGradient(smooth_img, mode='Sobel')
# T_high, T_low = lena.FindThreshold(magnitude)
# supressed = lena.NonmaximaSupress(magnitude, direction)
# edge_link = lena.EdgeLinking(supressed, T_high, T_low)
#
# test1 = Solution('images/test1.bmp', 'test1')
# smooth_img = test1.GaussSmoothing([3,3], 1)
# magnitude, direction = test1.ImageGradient(smooth_img, mode='Sobel')
# T_high, T_low = test1.FindThreshold(magnitude)
# supressed = test1.NonmaximaSupress(magnitude, direction)
# edge_link = test1.EdgeLinking(supressed, T_high, T_low)
#
# joy1 = Solution('images/joy1.bmp', 'joy1')
# smooth_img = joy1.GaussSmoothing([3,3], 1)
# magnitude, direction = joy1.ImageGradient(smooth_img, mode='Sobel')
# T_high, T_low = joy1.FindThreshold(magnitude)
# supressed = joy1.NonmaximaSupress(magnitude, direction)
# edge_link = joy1.EdgeLinking(supressed, T_high, T_low)
#
# pointer1 = Solution('images/pointer1.bmp', 'pointer1')
# smooth_img = pointer1.GaussSmoothing([3,3], 1)
# magnitude, direction = pointer1.ImageGradient(smooth_img, mode='Sobel')
# T_high, T_low = pointer1.FindThreshold(magnitude)
# supressed = pointer1.NonmaximaSupress(magnitude, direction)
# edge_link = pointer1.EdgeLinking(supressed, T_high, T_low)

# lena = Solution('images/lena.bmp', 'lena')
# smooth_img = lena.GaussSmoothing([3,3], 1)
# smooth_img = lena.GaussSmoothing([3,3], 3)
# smooth_img = lena.GaussSmoothing([3,3], 10)

# smooth_img = lena.GaussSmoothing([1,1], 1)
# smooth_img = lena.GaussSmoothing([3,3], 1)
# smooth_img = lena.GaussSmoothing([9,9], 1)

# lena = Solution('images/lena.bmp', 'lena')
# smooth_img = lena.GaussSmoothing([3,3], 1)
# magnitude, direction = lena.ImageGradient(smooth_img, mode='Sobel')
# T_high, T_low = lena.FindThreshold(magnitude, percentageOfNonEdge= 0.8)
# supressed = lena.NonmaximaSupress(magnitude, direction)
# edge_link = lena.EdgeLinking(supressed, T_high, T_low)
#
# T_high, T_low = lena.FindThreshold(magnitude, percentageOfNonEdge= 0.5)
# supressed = lena.NonmaximaSupress(magnitude, direction)
# edge_link = lena.EdgeLinking(supressed, T_high, T_low)
