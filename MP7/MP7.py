import cv2 as cv
import numpy as np
import os
from os.path import isfile, join

class Solution():
    def __init__(self, path, title):
        # image
        self.path = path
        # just the first image in the folder
        self.img = cv.imread(path + '0001.jpg', cv.IMREAD_GRAYSCALE)
        cv.imwrite('Results/original.jpg', self.img)

        if self.img is None:
            raise NameError('image not found')

        self.title = title

    def createVideo(self, title, path, fps):
        height, width = self.img.shape
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video=cv.VideoWriter(title + '.mov', fourcc, fps, (width,height))

        # note this length and string format is just for our video input
        for i in range(501):
            # images are formatted 0001 to 0500
            if len(str(i)) == 1:
                num = '000' + str(i)
            elif len(str(i)) == 2:
                num = '00' + str(i)
            else:
                num = '0' + str(i)
            img = cv.imread(path + num + '.jpg')

            # write image to video
            video.write(img)

        cv.destroyAllWindows()
        video.release()

    def targetTracking(self, mode='SSD'):
        ## create initial image boundary ##
        # var to store points of rectangle
        points = []

        # define click event Function
        def click_event(event, x, y, flags, params):
            # connect to points
            nonlocal points

            # create display image
            display = self.img.copy()

            # check for left click
            if event == cv.EVENT_LBUTTONDOWN:
                # add to points
                points = [(x,y)]

                # display coords
                print('p1:', x, ' ', y)

            # check for left click release
            elif event==cv.EVENT_LBUTTONUP:
                # add to points
                points.append((x,y))

                # display coords
                print('p2:', x, ' ', y)

                # display box
                cv.rectangle(display, points[0], points[1], (255, 0, 0), 1)
                cv.imshow("Testing Image", display)

        # display image
        cv.imshow('Testing Image', self.img)

        # use click_event to produce area
        cv.setMouseCallback('Testing Image', click_event)

        # wait for a key to be pressed to destroy all windows
        cv.waitKey(0)
        cv.destroyAllWindows()

        # crop image
        if len(points) == 2:
            min_y = min(points[0][1], points[1][1])
            max_y = max(points[0][1], points[1][1])
            min_x = min(points[0][0], points[1][0])
            max_x = max(points[0][0], points[1][0])
            T = self.img[min_y:max_y, min_x:max_x]
            cv.imshow("Template image", T)
            cv.imwrite('Results/' + mode + '/' + 'cropped_img.jpg', T)
            cv.waitKey(0)

        ## creating BoundingBox for each new image ##
        height, width = self.img.shape

        # template image
        t_height, t_width = T.shape

        # search window margins (one third total)
        window_margin_height = height // 6
        window_margin_width = width // 6

        # for each of the 500 images
        for s in range(1,501):
            # images are formatted 0001 to 0500
            if len(str(s)) == 1:
                num = '000' + str(s)
            elif len(str(s)) == 2:
                num = '00' + str(s)
            else:
                num = '0' + str(s)
            name = self.path + num + '.jpg'

            # create resultant image and testing grayscaled image
            result = cv.imread(name)
            test = cv.imread(name, cv.IMREAD_GRAYSCALE)

            # create search window margins
            window_points = [(points[0][0]-window_margin_width, points[0][1]-window_margin_height),(points[1][0]+window_margin_width, points[1][1]+window_margin_height)]

            # draw search window boundary
            cv.rectangle(result, window_points[0], window_points[1], (0,255,0), 1)
            cv.putText(result, 'search window', window_points[0], cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

            # max_val if CC or NCC, min_val if SSD
            max_val = float('-inf')
            min_val = float('inf')
            object_points = [[],[]]

            # exhaustive search through all 500 images
            for y in range(max(0, window_points[0][1]), min(height, window_points[1][1]-1)):
                for x in range(max(0, window_points[0][0]), min(width, window_points[1][0]-1)):
                    if y + t_height in range(height) and x + t_width in range(width):
                        I = test[y:y+t_height, x:x+t_width]
                        if mode == 'SSD':
                            val = self.SSD(I, T)
                            if min_val > val:
                                min_val = val
                                object_points = [(x,y),(x+t_width,y+t_height)]
                                new_T = I
                        else:
                            if mode == 'CC':
                                val = self.CC(I, T)
                            else:
                                val = self.NCC(I, T)

                            if max_val < val:
                                max_val = val
                                object_points = [(x,y),(x+t_width,y+t_height)]
                                new_T = I

            # set new template and window points
            T = new_T
            points = object_points

            # draw image boundary
            cv.rectangle(result, points[0], points[1], (255,0,0), 1)
            cv.putText(result, 'object', points [0], cv.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)

            # save image
            cv.imwrite('Results/' + mode + '/' + num + '.jpg', result)
            print(s)
            # cv.imshow('window', result)
            # cv.waitKey(0)


    ## Metrics ## (could be seperate class?)
    # sum of squared diff
    def SSD(self, I, T):
        total = 0

        diff = I - T
        total = np.sum(diff**2)

        return total

    # cross-correlation
    def CC(self, I, T):
        total = 0

        mult = I*T
        total = np.sum(mult)

        return total

    # normalized cross-correlation
    def NCC(self, I, T):
        total = 0

        I_mean = np.mean(I)
        T_mean = np.mean(T)

        I_norm = I - I_mean
        T_norm = T - T_mean

        numer = np.sum(I_norm * T_norm)
        denom = np.sqrt(np.sum(I_norm**2) * np.sum(T_norm**2))
        total = numer/denom
        return total

# Experiment
girl = Solution('image_girl/','girl')
# girl.createVideo('original', girl.img_path, 15)
# girl.targetTracking(mode='NCC')
# girl.createVideo('tracked_NCC', 'Results/NCC/', 15)
# girl.targetTracking(mode='SSD')
# girl.createVideo('tracked_SSD', 'Results/SSD/', 15)
girl.targetTracking(mode='CC')
girl.createVideo('tracked_CC', 'Results/CC/', 15)
