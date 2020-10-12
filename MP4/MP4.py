import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Solution():
    def __init__(self):
        # image trained on
        self.img_test = None
        # image tested on
        self.img_train = None
        # cropped training image
        self.crop = None
        # skin color histogram (HS)
        self.Skin_H = {}

    # function takes in an image that has human skin color, lets you crop image,
    # and stores (H,S) data as a Histogram
    def train(self, path, mode = "HSV"):
        self.img_test = cv.imread(path)

        # var to store points of rectangle
        points = []

        # define click event Function
        def click_event(event, x, y, flags, params):
            # connect to points
            nonlocal points

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
                cv.rectangle(self.img_test, points[0], points[1], (255, 0, 0), 1)
                cv.imshow("Testing Image", self.img_test)

        # display image
        cv.imshow('Testing Image', self.img_test)

        # use click_event to produce area
        cv.setMouseCallback('Testing Image', click_event)

        # wait for a key to be pressed to destroy all windows
        cv.waitKey(0)
        cv.destroyAllWindows()

        # crop image
        if len(points) == 2:
            min_x = min(points[0][1], points[1][1])
            max_x = max(points[0][1], points[1][1])
            min_y = min(points[0][0], points[1][0])
            max_y = max(points[0][0], points[1][0])
            self.crop = self.img_test[min_x+1:max_x, min_y+1:max_y]
            cv.imshow("cropped image", self.crop)
            cv.waitKey(0)

        # if mode is HSV, turn RGB to HSV
        if mode == "HSV":
            self.crop = cv.cvtColor(self.crop, cv.COLOR_BGR2HSV)
            cv.imshow("HSV of shopped image", self.crop)
            cv.waitKey(0)

            # fill histogram
            crop_height, crop_width, _ = self.crop.shape
            graph_x = []
            graph_y = []

            m = (0, ('',''))
            for i in range(crop_height):
                for j in range(crop_width):
                    (H, S) = self.crop[i,j][0:2]
                    self.Skin_H[(H,S)] = self.Skin_H.get((H,S),0) + 1

                    # for graphing
                    graph_x.append(H)
                    graph_y.append(S)

                    # to keep track of max
                    if self.Skin_H[(H,S)] > m[0]:
                        m = (self.Skin_H[(H,S)], (H,S))

            # create histogram graph
            fig = plt.figure()
            plt.hist2d(graph_x, graph_y, bins=(20,20))
            plt.xlabel('H')
            plt.ylabel('S')
            plt.colorbar()
            fig.suptitle('Histogram of pixel colors (H,S)')
            fig.savefig('Results/2D H,S Histogram.png')

            # normalize histogram
            for key in self.Skin_H:
                self.Skin_H[key] = self.Skin_H[key]/m[0]

        elif mode == "RGB":
            # self.crop is already RGB by default
            # cv.imshow("RGB of shopped image", self.crop)
            # cv.waitKey(0)

            # fill histogram
            crop_height, crop_width, _ = self.crop.shape
            graph_x = []
            graph_y = []

            m = (0, ('R','G'))
            for i in range(crop_height):
                for j in range(crop_width):
                    (R,G) = self.crop[i,j][0:2]
                    self.Skin_H[(R,G)] = self.Skin_H.get((R,G),0) + 1

                    # for histogram
                    graph_x.append(R)
                    graph_y.append(G)

                    # to keep track of max
                    if self.Skin_H[(R,G)] > m[0]:
                        m = (self.Skin_H[(R,G)], (R,G))

            # create histogram graph
            fig = plt.figure()
            plt.hist2d(graph_x, graph_y, bins=(20,20))
            plt.xlabel('R')
            plt.ylabel('G')
            plt.colorbar()
            fig.suptitle('Histogram of pixel colors (R,G)')
            fig.savefig('Results/2D R,G Histogram.png')

            # normalize histogram
            for key in self.Skin_H:
                self.Skin_H[key] = self.Skin_H[key]/m[0]


        elif mode == "N-RGB":
            # normalize self.crop
            self.crop = self.crop / 255
            # cv.imshow("RGB of shopped image", self.crop)
            # cv.waitKey(0)

            # fill histogram
            crop_height, crop_width, _ = self.crop.shape
            graph_x = []
            graph_y = []

            m = (0, ('R','G'))
            for i in range(crop_height):
                for j in range(crop_width):
                    (R,G) = self.crop[i,j][0:2]
                    self.Skin_H[(R,G)] = self.Skin_H.get((R,G),0) + 1

                    # for histogram
                    graph_x.append(R)
                    graph_y.append(G)

                    # to keep track of max
                    if self.Skin_H[(R,G)] > m[0]:
                        m = (self.Skin_H[(R,G)], (R,G))

            # create histogram graph
            fig = plt.figure()
            plt.hist2d(graph_x, graph_y, bins=(20,20))
            plt.xlabel('R')
            plt.ylabel('G')
            plt.colorbar()
            fig.suptitle('Histogram of pixel colors (NR,NG)')
            fig.savefig('Results/2D NR,NG Histogram.png')

            # normalize histogram
            for key in self.Skin_H:
                self.Skin_H[key] = self.Skin_H[key]/m[0]

    def test(self, path, threshold = .05, mode = 'HSV', title = ""):
        # set testing image
        self.img_test = cv.imread(path)

        # create result image
        res_height, res_width, _ = self.img_test.shape
        result_image = np.full((res_height, res_width, 3), 255, dtype=np.uint8)

        if mode == 'HSV':
            # convert img to HSV
            hsv_img = self.img_test.copy()
            hsv_img = cv.cvtColor(hsv_img, cv.COLOR_BGR2HSV)

            # fill result image w pixels from original image if pixel > threshold
            for i in range(res_height):
                for j in range(res_width):
                    (H,S) = hsv_img[i,j][0:2]
                    if (H,S) in self.Skin_H and self.Skin_H[(H,S)] > threshold:
                        result_image[i][j] = self.img_test[i][j]

            cv.imshow("result", result_image)
            cv.imwrite('Results/' + title + '.jpg', result_image)
            cv.waitKey(0)

        elif mode == 'RGB':
            # no need to convert
            rgb_img = self.img_test.copy()

            # fill result image w pixels from original image if pixel > threshold
            for i in range(res_height):
                for j in range(res_width):
                    (R,G) = rgb_img[i,j][0:2]
                    if (R,G) in self.Skin_H and self.Skin_H[(R,G)] > threshold:
                        result_image[i][j] = self.img_test[i][j]

            cv.imshow("result", result_image)
            cv.imwrite('Results/' + title + '.jpg', result_image)
            cv.waitKey(0)

        elif mode == 'N-RGB':
            # just need to normalize
            rgb_img = self.img_test.copy()/255

            # fill result image w pixels from original image if pixel > threshold
            for i in range(res_height):
                for j in range(res_width):
                    (R,G) = rgb_img[i,j][0:2]
                    if (R,G) in self.Skin_H and self.Skin_H[(R,G)] > threshold:
                        result_image[i][j] = self.img_test[i][j]

            cv.imshow("result", result_image)
            cv.imwrite('Results/' + title + '.jpg', result_image)
            cv.waitKey(0)

# experiment
soln = Solution()

## train based on testing_image_1 ##
# HSV
# soln.train('images/testing_image_1.jpg', mode="HSV")
# soln.test('images/gun1.bmp', threshold=.025, mode="HSV", title = "gun1 HSV")
# soln.test('images/joy1.bmp', mode="HSV", title="joy1 HSV")
# soln.test('images/pointer1.bmp', mode="HSV", title="pointer1 HSV")

# RGB
# soln.train('images/testing_image_1.jpg', mode="RGB")
# soln.test('images/gun1.bmp', threshold=0, mode="RGB", title="gun1 RGB")

# soln.train('images/joy1.bmp', mode="RGB")
# soln.test('images/gun1.bmp', threshold=.02, mode="RGB",title="gun1 RGB (joy1 train)")
# soln.test('images/pointer1.bmp', threshold=0.02, mode="RGB", title="pointer1 RGB (joy1 train)")

# soln.train('images/pointer1.bmp', mode="RGB")
# soln.test('images/joy1.bmp', threshold=.05, mode="RGB",title="joy1 RGB (pointer1 train)")

# N-RGB
# soln.train('images/joy1.bmp', mode="N-RGB")
# soln.test('images/gun1.bmp', threshold=.02, mode="N-RGB",title="gun1 N-RGB (joy1 train)")
# soln.test('images/pointer1.bmp', threshold=0.02, mode="N-RGB", title="pointer1 N-RGB (joy1 train)")
#
# soln.train('images/pointer1.bmp', mode="N-RGB")
# soln.test('images/joy1.bmp', threshold=.05, mode="N-RGB",title="joy1 N-RGB (pointer1 train)")
