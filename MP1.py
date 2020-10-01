import cv2 as cv
import collections

def CCL(path):
    # read image
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    # get image height and width
    height, width = img.shape

    # initialize variables
    num = 0
    label_img = []
    label = [[0 for i in range(width)] for j in range(height)]
    E_table = collections.defaultdict(int)

    # first pass
    for i in range(height):
        for j in range(width):
            # if pixel is white
            if img[i,j] == 255:
                # calculate labels of upper and left pixel w/ boundary conditions
                if i-1 in range(height):
                    upper = label[i-1][j]
                else:
                    upper = 0
                if j-1 in range(width):
                    left = label[i][j-1]
                else:
                    left = 0

                if upper == 0 and left == 0:
                    num += 1
                    label[i][j] = num
                elif left == 0 or upper == 0:
                    label[i][j] = max(upper, left)
                elif left == upper:
                    label[i][j] = upper
                else:
                    label[i][j] = min(upper, left)
                    E_table[max(upper, left)] = max(E_table[max(upper, left)], label[i][j])

    # second pass
    # set for holding label roots
    s = set()

    # helper function find_root -> this is basically union find
    def find_root(node):
        if node not in E_table:
            s.add(node)
            return node
        else:
            return find_root(E_table[node])

    # rearrange E_table to have all labels point to their roots
    for val in E_table:
        E_table[val] = find_root(E_table[val])

    # create true labels for each unique label that we've acquired
    ## size filter: for gun.bmp only ##
    areas = collections.defaultdict(int)
    ###                             ###

    dict = collections.defaultdict(int)
    for ind, num in enumerate(s):
        dict[num] = ind+1

    for i in range(height):
        for j in range(width):
            # if pixel is white
            # check E_table to see if it should be replaced
            if img[i,j] == 255 and label[i][j] in E_table:
                label[i][j] = E_table[label[i][j]]
                ## for size filter ##
                areas[label[i][j]] += 1
            else:
                areas[label[i][j]] += 1
                ###               ###

    ## for size filter ##
    min_area = 1000
    for key in areas:
        if areas[key] < min_area:
            areas[key] = -1
    ####

    # create label_img
    label_img = img
    for i in range(height):
        for j in range(width):
            if label[i][j]:
                # gun.bmp ONLY -> for size filter
                if areas[label[i][j]] == -1:
                    color = 0
                else:
                    color = 255/(len(s)+1) * dict[label[i][j]]
                label_img[i][j] = color

    # returning label_img doesn't do much, so i have the function show it as well
    title = (path.split('/')[1])[:-4] + 'size_filtered.jpg'
    cv.imwrite(title, label_img)
    cv.imshow(title, label_img)
    cv.waitKey(0)
    print('title:', title, 'num: ', len(s))
    return label_img, num

# CCL("images/test.bmp")
# CCL("images/face.bmp")
CCL("images/gun.bmp")
# CCL("images/face_old.bmp")
