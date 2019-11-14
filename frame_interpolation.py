import cv2
import sys
import numpy as np
import pickle
import numpy as np
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt

"""
Cole Cofer
CS410 Computer Vision
Portland State University

Image Interpolation
This program take a frame0 and frame1 and creates an interpolated image between them

Execution example:
python frame_interpolation.py frame0.png frame1.png flow0.flo frame05.png
"""

BLUR_OCC = 3
NEIGH_PIXELS = 8


def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def find_holes(flow):
    '''
    Find a mask of holes in a given flow matrix
    Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
    :param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
    :return: a mask annotated 0=hole, 1=no hole
    '''

    h, w, _ = flow.shape
    hole_matrix = np.full((h,w), 1)
    #Iterate through flow vector and check for holes
    for y in range(h):
        for x in range(w):
            u = abs(flow[y, x][0])
            v = abs(flow[y, x][1])
            if u > 10e9 or np.isnan(u) or np.isinf(u) or v >= 10e9 or np.isnan(v) or np.isinf(v):
                hole_matrix[y, x] = 0

    return hole_matrix


def distBetweenPoints(pt1, pt2):
    ''' Returns the distance between two points passed as tuples '''
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def holefill(flow, holes):
    '''
    fill holes in order: row then column, until fill in all the holes in the flow
    :param flow: matrix of dense optical flow, it has shape [h,w,2]
    :param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
    :return: flow: updated flow
    '''
    h_hole, w_hole = holes.shape
    h_flow, w_flow, _ = flow.shape
    stillHasHoles = True

    while (stillHasHoles == True):
        stillHasHoles = False
        for y in range(h_hole):
            for x in range(w_hole):
                if (holes[y][x] == 0):
                    neighborhood = []
                    #Top row
                    if y+2 <= h_hole:
                        if x-1 >= 0 and holes[y+1][x-1] != 0: neighborhood.append(flow[y+1][x-1])
                        if holes[y+1][x] != 0: neighborhood.append(flow[y+1][x])
                        if x+2 < w_hole and holes[y+1][x+1] != 0: neighborhood.append(flow[y+1][x+1])
                    #Middle row
                    if x-1 >= 0 and holes[y][x-1] != 0: neighborhood.append(flow[y][x-1])
                    if x+2 <= w_hole and holes[y][x+1] != 0: neighborhood.append(flow[y][x+1])
                    #Bottom row
                    if y-1 >= 0:
                        if x-1 >= 0 and holes[y-1][x-1] != 0: neighborhood.append(flow[y-1][x-1])
                        if holes[y-1][x] != 0: neighborhood.append(flow[y-1][x])
                        if x+2 < w_hole and holes[y-1][x+1] != 0: neighborhood.append(flow[y-1][x+1])

                    #Calculate average for pixels in neighborhood that contain values
                    u_avg, v_avg = 0, 0
                    validInNeighborhood = len(neighborhood)
                    for pixel in neighborhood:
                        u_avg += pixel[1]
                        v_avg += pixel[0]

                    #Only update if you have data in the neighborhood
                    if validInNeighborhood > 0:
                        flow[y][x] = (v_avg / validInNeighborhood, u_avg / validInNeighborhood)
                        holes[y][x] = 1
                    else:
                        stillHasHoles = True

    return flow

def occlusions(flow0, frame0, frame1):
    '''
    Follow the step 3 in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
    :param flow0: dense optical flow
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :return:
    '''

    height, width, _ = flow0.shape
    occ0 = np.zeros([height,width],dtype=np.float32)
    occ1 = np.zeros([height,width],dtype=np.float32)

    # ==================================================
    # ===== step 4/ warp flow field to target frame
    # ==================================================
    flow1 = interpflow(flow0, frame0, frame1, 1.0)
    pickle.dump(flow1, open('flow1.step4.data', 'wb'))
    # ====== score

    flow1       = pickle.load(open('flow1.step4.data', 'rb'))
    flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
    diff = np.sum(np.abs(flow1-flow1_step4))
    print('flow1_step4: ',diff)

    # ==================================================
    # ===== main part of step 5
    # ==================================================

    h, w, _ = flow0.shape
    for y in range(h):
        for x in range(w):
            u, v = flow0[y][x][0], flow0[y][x][1]
            u1, v1 = flow1[y][x][0], flow1[y][x][1]

            newx = int(round(u+x))
            newy = int(round(v+y))
            if newx >= w or newy >= h or newx < 0 or newy < 0:
                occ1[y][x] = 1
                if newx == -1:
                    occ1[y][x] = 0
            else:
                diff = np.sum(abs(flow0[y][x] - flow1[newy][newx]))
                if diff > 0.5:
                    occ1[y][x] = 1
                    if u <= -3.5:
                        occ1[y][x] = 0

            if u1 >= 1.0e9 or np.isnan(u1) or np.isinf(u1) or v1 >= 1.0e9 or np.isnan(v1) or np.isinf(v1):
                occ0[y][x] = 1

    return occ0, occ1


class MappingList:
    '''
    Holds all pixels that want to map to the same place from fram0 to frame1,
    as well as the vector that they points to (collisionFlow).
    This allows you to add the collision point with the collisionFlow point
    together to get the RGB value in frame1 to compare with and find the best match.

    frame0collisions => Origin point that gets mapped to frame1
    frametcollisionflow => flow[y,x] where (y,x) is the point in frame0collisions
    '''
    def __init__(self, frame0Pixels, frame1Pixels):
        self.frame0Pixels = frame0Pixels
        self.frame1Pixels = frame1Pixels

def round_px(x):
    ''' Return a normalized / rounded version of the passed in number '''
    return (int(roundX(x[0])), int(roundX(x[1])))

def floor_px(x):
    ''' Return the floor of both pixel values '''
    return (int(np.floor(x[0])), int(np.floor(x[1])))

def sumPixels(a, b, round):
    ''' Sums two pixels as tuples and rounds if specified '''
    sum = tuple(np.add(a, b))
    if round == 'round':
        return round_px(sum)
    elif round == 'floor':
        return floor_px(sum)
    else:
        return sum

def subtractPixels(a, b, round):
    ''' Sums two pixels as tuples and rounds if specified '''
    sum = tuple(np.subtract(a, b))
    if round == True:
        return round_px(sum)
    else:
        return sum

def checkNeighbor(x):
    '''
    Checks if when subtracting the radius to get the splatted pixels, that if it
    falls on a whole number then we want to take the pixel to the left or below it.
    '''
    if x.is_integer():
        return x-1
    else:
        return x

def roundX(x):
    '''
    When comparing with the TA's results it seems that using
    np.round or the python round does not give good results because
    they even numbers up and odd down which is atypical.
    '''
    if np.abs(x % 1) >= 0.5:
        if x >= 0: return np.ceil(x)
        return np.floor(x)
    else:
        if x >= 0: return np.floor(x)
        return np.ceil(x)

def findSplattingPixels(px, py, h, w, r, t, flow):
    '''
    Returns a list of up to six points if they are all within bounds of h*w
    Note that the points are returned as (x, y)
    '''

    adjacentPixels = []
    u, v = flow[py, px]

    exactCenterPixel = sumPixels((px, py), (u, v), round='False') #Exact location of the center we want to splat around
    cx, cy = exactCenterPixel
    neighborhood = [(checkNeighbor(cx-r), cy+r),                (cx, cy+r),                (cx+r, cy+r),
                    (checkNeighbor(cx-r), cy),                  (cx, cy),                  (cx+r, cy),
                    (checkNeighbor(cx-r), checkNeighbor(cy-r)), (cx, checkNeighbor(cy-r)), (cx+r, checkNeighbor(cy-r))]

    #Take the floor of all pixels in the neighborhood
    for px in neighborhood:
        adjacentPixels.append(floor_px(px))

    #Remove and duplicates
    adjacentPixels = set(adjacentPixels)

    return adjacentPixels


def interpflow(flow, frame0, frame1, t):
    '''
    Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
    Follow the algorithm (1) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param flow: dense optical flow from frame0 to frame1
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param t: the intermiddite position in the middle of the 2 input frames
    :return: a warped flow
    '''

    h, w, _ = frame0.shape

    iflow = [[(10e10, 10e10) for j in range(w)] for i in range(h)]
    r = 0.5

    #Create mappingList to hold collisions
    mappingList = [[MappingList([], []) for j in range(w*2)] for i in range(h*2)] #TODO We should probably remove h*2/w*2 and do bounds checking

    #Forward ward pixel from frame0 to framet (when t = 1)
    #Perform splatting and append each pixel in the splat into the collisions list
    #Also append the flow for the origin pixel
    for y in range(h):
        for x in range(w):
            splatPixels = findSplattingPixels(x, y, h, w, r, t, flow)
            for p in splatPixels:
                #Insert into the mapping list using t * p to get the pixel in framet
                px, py = round_px(np.dot(t, p))
                mappingList[py][px].frame0Pixels.append((y, x))        #Append the frame0 point
                mappingList[py][px].frame1Pixels.append((p[1], p[0]))  #Append the frame1 point

    #Now that we have a list of potential candidates and can determine the best matches
    #Note: This could be optimized by doing this all at once in the above step...
    #However it doesn't seem any slower than other students.
    for y in range(h):
        for x in range(w):
            mappings = mappingList[y][x]
            numPixelCandidates = len(mappings.frame0Pixels)

            #Check if there is only one mapping (meaning it's the best match)
            if numPixelCandidates == 1:
                #Get the index for the origin point so that we can insert a (x, y) pair into iflow,
                #and have it give us where that point maps too
                frame0y, frame0x = mappings.frame0Pixels[0][0], mappings.frame0Pixels[0][1]
                iflow[frame0y][frame0x] = (y, x)

            elif numPixelCandidates > 1:
                #Store the current best match (index, colorDiff)
                bestMatch = [0, 10e10]

                #Iterate through potential pixel candidates and find the best match
                for i in range(numPixelCandidates):
                    frame0y, frame0x = mappings.frame0Pixels[i][0], mappings.frame0Pixels[i][1]
                    frame1y, frame1x = mappings.frame1Pixels[i][0], mappings.frame1Pixels[i][1]

                    if inWarpBounds(frame0y, frame0x, h, w) and inWarpBounds(frame1y, frame1x, h, w):
                        diff = calcRGBDiff(frame0[frame0y][frame0x], frame1[frame1y][frame1x])

                        #Find the warping with the least color difference
                        if diff < bestMatch[1]:
                            bestMatch[0] = i
                            bestMatch[1] = diff

                #Add the best match into iflow
                bestIndex = bestMatch[0]
                frame0y, frame0x = mappings.frame0Pixels[bestIndex][0], mappings.frame0Pixels[bestIndex][1] #Gives you the origin pixel of the best match
                iflow[frame0y][frame0x] = (y, x)

    return np.asarray(iflow)


def calcRGBDiff(a, b):
    ''' Returns the difference of two RGB tuples '''
    return np.sum(np.abs(np.subtract(a, b)))


def inBounds(y, x, h, w):
    ''' Checks if a point (y, x) is within bounds of a .5px radius '''
    if y >= 0 and x >= 0 and y <= h and x <= w:
        return True
    else:
        return False


def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''
    iframe = np.zeros_like(frame0).astype(np.float32)
    h, w, _ = iframe.shape

    #Iterate through interpolated image
    for y in range(h):
        for x in range(w):
            #Calculate (y,x) coordinates for frame0 and frame1
            X = [x, y]

            X0 = subtractPixels(X, np.dot(t, iflow[y][x]), round=True)
            X1 = sumPixels(X, np.dot((1-t), iflow[y][x]), round='round')

            X0x, X0y = X0[0], X0[1]
            X1x, X1y = X1[0], X1[1]

            #Check if the pixel is visible in both images, or only one
            if inWarpBounds(X0y, X0x, h, w) == True and inWarpBounds(X1y, X1x, h, w) == True:
                #Get occlusion values and round them
                occ0Value = int(roundX(occ0[X0y][X0x]))
                occ1Value = int(roundX(occ1[X1y][X1x]))
                if occ0Value == 0 and occ1Value == 0:
                    iframe[y][x] = np.add(np.dot((1 - t), frame0[X0y][X0x]), np.dot(t, frame1[X1y][X1x]))
                elif occ0Value == 1 and occ1Value == 0:
                    iframe[y][x] = frame1[X1y][X1x]
                elif occ0Value == 0 and occ1Value == 1:
                    iframe[y][x] = frame0[X0y][X0x]

            #If both values are not in bounds, then at least use the coordinate that is inbounds and then
            #grab the closest pixel to the edge of the image
            elif inWarpBounds(X0y, X0x, h, w) == True and inWarpBounds(X1y, X1x, h, w) == False:
                occ0Value = int(roundX(occ0[X0y][X0x]))
                iframe[y][x] = frame1[X0y][X0x]

            elif inWarpBounds(X0y, X0x, h, w) == False and inWarpBounds(X1y, X1x, h, w) == True:
                occ1Value = int(roundX(occ1[X1y][X1x]))
                iframe[y][x] = frame0[X1y][X1x]

    return iframe


def inWarpBounds(y, x, h, w):
    ''' Checks if a point (y, x) is within bounds of h * w '''
    if y >= 0 and x >= 0 and y < h and x < w:
        return True
    else:
        return False

def blur(im):
    '''
    blur using a gaussian kernel [5,5] using opencv function: cv2.GaussianBlur, sigma=0
    :param im:
    :return updated im:
    '''
    im = cv2.GaussianBlur(im,(5,5),0)

    return im

def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=',str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0,open('holes0.step2.data','wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data','rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes0-holes0_step2))
    print('holes0_step2',diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0,holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb')) # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb')) # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow0-flow0_step3))
    print('flow0_step3',diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0,frame0,frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb')) # save your intermediate result
    # ===== score

    # occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    # occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5',diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5',diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================

    for iblur in range(0,BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)

    pickle.dump(occ0, open('occ0.step6.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb')) # save your intermediate result
    # ===== score

    occ0        = pickle.load(open('occ0.step6.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb')) # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6',diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6',diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
    flow_t = interpflow(flow0, frame0, frame1, t)
    pickle.dump(flow_t, open('flow_t.step7.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step7.data', 'rb')) # load your intermediate result
    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step7))
    print('flow_t_step7',diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
    holes1 = find_holes(flow_t)
    pickle.dump(holes1, open('holes1.step8.data', 'wb')) # save your intermediate result
    # ====== score
    holes1       = pickle.load(open('holes1.step8.data','rb')) # load your intermediate result
    holes1_step8 = pickle.load(open('holes1.step8.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes1-holes1_step8))
    print('holes1_step8',diff)

    # ===== fill in any hole in flow_t using an outside-in strategy
    flow_t = holefill(flow_t, holes1)
    pickle.dump(flow_t, open('flow_t.step8.data', 'wb')) # save your intermediate result
    # ====== score

    flow_t       = pickle.load(open('flow_t.step8.data', 'rb')) # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step8))
    print('flow_t_step8',diff)

    # ==================================================
    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
    # ==================================================
    frame_t = warpimages(flow_t, frame0, frame1, occ0, occ1, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb')) # save your intermediate result
    # ====== score

    frame_t       = pickle.load(open('frame_t.step9.data', 'rb')) # load your intermediate result
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb')) # load sample result
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32)-frame_t_step9.astype(np.float32))))
    print('frame_t',diff)

    return frame_t


if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    path_file_image_0 = sys.argv[1]
    path_file_image_1 = sys.argv[2]
    path_file_flow    = sys.argv[3]
    path_file_image_result = sys.argv[4]

    # ===== read 2 input images and flow
    frame0 = cv2.imread(path_file_image_0)
    frame1 = cv2.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)

    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
    cv2.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))
