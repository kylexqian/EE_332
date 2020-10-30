import cv2 as cv
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

class Solution():
    def __init__(self, path, title):
        # image
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if self.img is None:
            raise NameError('image not found')

        self.title = title

    def hough(self, theta_res, rho_res, threshold):
        ## edge detection ##
        # 60 and 120 are arbitrary
        edges = cv.Canny(self.img, 60, 120) #self.edge_detection(self.img, 60, 120)

        # cv.imshow('edge detection', edges)
        # cv.imwrite('Results/edge' + self.title + '.png', edges)
        # cv.waitKey(0)

        ## transform from xy space into parameter space ##
        height, width = self.img.shape

        # range of values for p based on quantization
        thetas = np.radians(np.linspace(-90,90, theta_res))
        max_dist = int(np.sqrt(height**2 + width**2))
        rhos = np.linspace(-max_dist, max_dist, max_dist*2)#rho_res)

        # caching values
        cos = np.cos(thetas)
        sin = np.sin(thetas)
        num_thetas = len(thetas)

        # create parameter space based off limits of theta and p
        param_space = np.zeros((num_thetas, 2*max_dist))

        # helper functio for xy2parameterspace
        def xy2parameterspace(x, y):
            # calculate rho for each theta
            for theta in range(num_thetas):
                rho = int(round(x*cos[theta] + y*sin[theta]) + max_dist)
                param_space[theta][rho] += 1

        for i in range(height):
            for j in range(width):
                if edges[i][j] > 0:
                    xy2parameterspace(i, j)

        # viewing
        param_space_img = param_space/np.max(param_space)*255
        param_space_img = param_space_img.astype(np.uint8)
        cv.imshow('param_space', param_space_img)
        # cv.imwrite('Results/parameter_space ' + self.title + ' ' + 'theta_res=' + str(theta_res) + ' rho_res=' + str(rho_res) + '.png', param_space_img)
        cv.waitKey(0)

        ## thresholding ##
        height, width = param_space.shape
        max_num = np.max(param_space)
        thresh_img = np.zeros(param_space.shape)

        for i in range(height):
            for j in range(width):
                if int(param_space[i][j]) >= threshold*max_num:
                    thresh_img[i][j] = 255

        cv.imshow('threshold', thresh_img)
        cv.imwrite('Results/threshold ' + self.title + ' ' + 'theta_res=' + str(theta_res) + ' rho_res=' + str(rho_res) + '.png', thresh_img)
        cv.waitKey(0)

        ## Finding maximas ##
        # arbitrary values
        filter_size = 10 # for filter size
        threshold = 50 # for difference threshold # 40 for input.bmp, 50 else

        # filter max and min based on filter_size
        data_max = filters.maximum_filter(param_space, filter_size)
        data_min = filters.minimum_filter(param_space, filter_size)

        # find when difference between max and min is greater than some threshold
        maxima = (param_space == data_max)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        # CCL
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)

        # get center of each slice
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2
            y.append(y_center)

        maximas = np.zeros(param_space.shape)
        for i in range(len(x)):
            maximas[int(y[i]),int(x[i])] = 255

        cv.imshow('maximas', maximas)
        cv.imwrite('Results/non-maxima supressed for ' + self.title + '.png', maximas)
        cv.waitKey(0)

        ## revert maximas in parameter space back to xy space ##
        print(y)
        print(x)
        # print(param_space.shape)
        final = edges
        height, width = self.img.shape
        print('height', height)
        print('width', width)
        for i, j in zip(y, x):
            theta = thetas[int(i)]
            rho = rhos[int(j)]
            deg = np.degrees(theta)

            if deg >= -45 and deg <= 45:
                print('thin')
                for y in range(height):
                    x = int((-y*np.sin(theta) + rho)/np.cos(theta))
                    if x in range(width):
                        print(y,x)
                        final[y][x] = 255
            else:
                print('thick')
                for x in range(width):
                    y = int((-x*np.cos(theta)+rho)/np.sin(theta))
                    # print(y, x)
                    if y in range(height):
                        print(y,x)
                        final[y][x] = 255
        # cv.line(final, (0, 0), (1, 1), (255,0,0), thickness = 10)
        # cv.imshow('final', final)
        # cv.waitKey(0)

        cv.imshow('final', self.img)
        cv.waitKey(0)

# [79.0, 218.0, 226.0, 353.0]
# [291.0, 569.0, 453.0, 407.0]

# [38.0, 47.0, 174.0, 258.0]
# [569.0, 453.0, 406.0, 432.0]

# experiment
# soln = Solution('images/test.bmp', 'test')
# soln.hough(360, 1000, threshold = 0.5)
# #
# soln = Solution('images/test2.bmp', 'test2')
# soln.hough(180, 1000, threshold = 0.5)

# soln = Solution('images/input.bmp', 'input')
# soln.hough(360, 100, threshold = 0.5)
