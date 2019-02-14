import cv2
import numpy as np,sys
import math

def mutual_information(a,b):

    hgram = cv2.calcHist( [a, b], [0, 1], None, [256, 256], [0, 256, 0, 256] )

    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def getPyramid (A):
    print('getPyramid')
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]

    loop = True

    while(loop == True):
        G = cv2.pyrDown(G)
        gpA.append(G)
        r,c = G.shape
        if r < 100 or c < 100:
            loop = False
    # generate Laplacian Pyramid for A
    lpA = [gpA[len(gpA)-1]]
    for i in range(len(gpA)-1,0,-1):

        size = (gpA[i-1].shape[1],gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i],dstsize=size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)
    return gpA, lpA


def matchImage(a,b,poi,poj,r=1):
    print('matchImage ',poi,' ',poj)

    #cv2.imshow('a',a)
    #cv2.imshow('b',b)



    #cv2.waitKey()

    #print('matching...')


    rows,cols = a.shape
    result = {}

    rr = a - b
    rr *= rr

    best = mutual_information(a,b)
    best_oi = 0
    best_oj = 0
    tmp_best = 0

    for oi in range(-r,r+1,1):
        for oj in range(-r,r+1,1):
            B_r = np.zeros((rows,cols,1), np.uint8)

            B_r_i1 = max(0,oi+poi)
            B_r_i2 = min(rows+oi+poi,rows)
            B_r_j1 = max(0,oj+poj)
            B_r_j2 = min(cols+oj+poj,cols)

            B_i1 = max(0,-oi-poi)
            B_i2 = min(rows-oi-poi,rows)
            B_j1 = max(0,-oj-poj)
            B_j2 = min(cols-oj-poj,cols)

            B_r = b[B_i1:B_i2, B_j1:B_j2]
            A_r = a[B_r_i1:B_r_i2, B_r_j1:B_r_j2]


            tmp_best = mutual_information(A_r,B_r)
            if tmp_best > best:
                best = tmp_best
                best_oi = oi
                best_oj = oj

    return best_oi,best_oj,best



def matchPyramid(gA,gB):
    print('patchPiramid')
    oi = 0
    oj = 0
    score = 0
    for i in range(len(gA)-1,0,-1):
        noi,noj,score = matchImage(gA[i],gB[i],oi,oj,1)
        noi += oi
        noj += oj
        if i != 0:
            oi = 2*noi
            oj = 2*noj
        else:
            oi = noi
            oj = noj

    return oi,oj,score




# A is the origin image, B is to readjust
def readjustment (A,B):
    print('readjustement')

    gpA, lpA = getPyramid(A)
    gpB, lpB = getPyramid(B)

    oi,oj,score = matchPyramid(gpA,gpB)
    print('final move : ',oi,';',oj)
    rows,cols = A.shape
    new_B = np.zeros((rows,cols,1), np.uint8)
    rows,cols = B.shape
    new_B.shape = B.shape

    new_B_i1 = max(0,oi)
    new_B_i2 = min(rows+oi,rows)
    new_B_j1 = max(0,oj)
    new_B_j2 = min(cols+oj,cols)

    B_i1 = max(0,-oi)
    B_i2 = min(rows-oi,rows)
    B_j1 = max(0,-oj)
    B_j2 = min(cols-oj,cols)

    new_B[new_B_i1:new_B_i2, new_B_j1:new_B_j2] = B[B_i1:B_i2, B_j1:B_j2]

    return new_B,score,oi,oj

def getBGR(im):
    print('getBGR')
    print (im.shape)
    rows,cols = im.shape
    _rows = math.floor(rows/3.0)
    B = im[0:_rows,0:cols]
    G = im[_rows:_rows*2,0:cols]
    R = im[_rows*2:_rows*3,0:cols]

    return B,G,R

def crop(img,Bi,Bj,Ri,Rj):
    left_c = 0
    right_c = 0
    top_c = 0
    bottom_c = 0

    rows, cols, channels = img.shape

    if Bi > 0 and Ri > 0:
        top_c = max(Bi,Ri)
    elif Bi > 0 and Ri < 0:
        top_c = Bi
        bottom_c = Ri
    elif Bi < 0 and Ri > 0:
        top_c = Ri
        bottom_c = Bi
    elif Bi < 0 and Ri < 0:
        bottom_c = min(Bi,Ri)

    if Bj > 0 and Rj > 0:
        left_c = max(Bj,Rj)
    elif Bj > 0 and Rj < 0:
        left_c = Bj
        right_c = Rj
    elif Bj < 0 and Rj > 0:
        left_c = Rj
        right_c = Bj
    elif Bj < 0 and Rj < 0:
        right_c = min(Bj,Rj)

    return img[top_c:rows-top_c+bottom_c+1,left_c:cols-left_c+right_c+1]



def buildColoredImage(path):
    print('buildColoredImage')

    im = cv2.imread(path,flags=cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('im',gim)
    #cv2.waitKey()

    B,G,R = getBGR(im)

    B,score1,Bi,Bj = readjustment(G,B)
    R,score2,Ri,Rj = readjustment(G,R)

    BGR = cv2.merge([B, G, R])

    cv2.imshow('test',BGR)
    cv2.waitKey()

    BGRc = crop(BGR,Bi,Bj,Ri,Rj)

    cv2.imwrite('test.png',BGR)
    cv2.imwrite('testc.png',BGRc)


if __name__ == '__main__':
    buildColoredImage('01887u.tif')
