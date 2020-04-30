import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt
import math

# function for getting transformed image using homography
def warpPerspective(src,h,x,y):
    h = np.linalg.inv(h)
    dst = np.zeros((x,y,3),dtype = np.uint8)
    for i in range(x):
        for j in range(y):
            X = (h[0,0]*i+h[0,1]*j+h[0,2])/(h[2,0]*i+h[2,1]*j+h[2,2])
            Y = (h[1,0]*i+h[1,1]*j+h[1,2])/(h[2,0]*i+h[2,1]*j+h[2,2])
            if X>=0 and X<src.shape[0] and Y>=0 and Y<src.shape[1]:
                dst[j,i] = src[math.floor(Y),math.floor(X)]
    return dst

def rotate(l, n):
    return np.roll(l,n,axis =0)

def SubMatrix(srcPt,dstPt):
    xw = srcPt[0]
    yw = srcPt[1]
    xc = dstPt[0]
    yc = dstPt[1]
    output = np.array([[xw,yw,1,0,0,0,-xc*xw,-xc*yw,-xc],
                       [0,0,0,xw,yw,1,-yc*xw,-yc*yw,-yc]])
    return output

# Function for get homograpy from src to dst   
def getHomography(src,dst):
    rowSet = []
    for i,j in zip(src,dst):
        rowSet.append(SubMatrix(i,j))
    A = np.vstack((rowSet[0],rowSet[1],rowSet[2],rowSet[3]))
    U, s, V = np.linalg.svd(A)
    H= V[-1, :]/V[-1,-1]
    H=np.reshape(H,(3,3))
    return H

def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

cap = cv2.VideoCapture('Tag2.mp4')
lena = cv2.imread('Lena.png')

K1 =np.array([[1406.08415449821,0,0],
             [2.20679787308599, 1417.99930662800,0],
             [1014.13643417416, 566.347754321696,1]])
K = K1.T

count = 0
while(cap.isOpened()):
    _, frame = cap.read()

    frameCopy = copy.deepcopy(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #plt.hist(gray.ravel(),256,[0,256]); plt.show()

    (thresh, im_bw) = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    squares = []
    
    for j,cnt in zip(hierarchy[0],contours):
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            if j[3] != -1:
                squares.append(cnt)
                
    overLayBase = copy.deepcopy(frameCopy)
    arCodeImage = copy.deepcopy(frameCopy)
    
    for sq in squares:
        pts1 = np.float32([list(i) for i in sq])
        pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])
        M = getHomography(pts1,pts2)
        #dst = warpPerspective(frameCopy,M,800,800)
        dst = cv2.warpPerspective(frameCopy,M,(800,800))
        dst0 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        
#        plt.hist(dst.ravel(),256,[0,256]); plt.show()
        arTag = cv2.resize(dst0, (8,8))
        (thresh, arTag) = cv2.threshold(arTag, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        detectedCnt, hierarchy = cv2.findContours(arTag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours=detectedCnt[0]
        for i in detectedCnt:
            if cv2.contourArea(i) > 4 and cv2.isContourConvex(i):
                contours=i
        [[min1,min0]] = np.amin(contours,axis = 0)
        [[max1,max0]] = np.amax(contours,axis = 0)
        orientationIndex = [[min0,min1],[max0,min1],[max0,max1],[min0,max1]]

        lenaCor = [[0,0],[lena.shape[0],0],[lena.shape[0],lena.shape[1]],[0,lena.shape[1]]]
        #orientationCor = {1:"TopLeft", 2:"BottomLeft", 3:"TopRight", 4:"BottomRight" }
        orientationCor = 0
        for i,ori in zip(orientationIndex,[0,1,2,3]):
            if arTag[i[0],i[1]] >= 200:
                orientationCor = ori
                break  
            
        # decoding ar tag
        arCode = []
        arNum = ""
        for i in [[3,4],[4,4],[4,3],[3,3]]:
            if arTag[i[0],i[1]] == 255:
                arCode.append(1)
            else:
                arCode.append(0)
        rotate(arCode,orientationCor)
        
        for i in arCode:
            arNum = str(i) + arNum
        arNum = int(arNum,2)
        
        ## for testing Display oriented Lena in square
        lenaOr = rotate(lenaCor,orientationCor)
        M = getHomography(np.float32(lenaCor),np.float32(lenaOr))
        lenaOriented = cv2.warpPerspective(lena,M,(512,512))
        
        ptsOr = rotate(pts1,orientationCor)
        M = getHomography(np.float32(lenaCor),ptsOr)
        linaOnImage = cv2.warpPerspective(lena, M, (frameCopy.shape[1],frameCopy.shape[0]))
        _, mask = cv2.threshold(linaOnImage, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.cvtColor(mask_inv,cv2.COLOR_BGR2GRAY)
        frame_bg = cv2.bitwise_and(overLayBase,overLayBase,mask = mask_inv)
        linaOnImage = cv2.add(linaOnImage,frame_bg)
        overLayBase = linaOnImage
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(arCodeImage,"#"+str(arNum),(sq[0][0],sq[0][1]), font, 1,(0,0,255),2,cv2.LINE_AA)
        
        orientedCor = np.array(lenaCor)*100 / 512
        H = getHomography(orientedCor,pts1)
        P =  projection_matrix(K,H)
      
        cubearr = np.array([[0,0,0,1],[100,0,0,1],[100,100,0,1],[0,100,0,1],
                            [0,0,100,1],[100,0,100,1],[100,100,100,1],[0,100,100,1]])
        cubePts = cubearr.T
        imgPts = np.matmul(P,cubePts)
        imgPtsT = np.int32(imgPts.T)
        for i in imgPtsT:
            pt = np.int32(i[0:2]/i[2])
            cv2.circle(frameCopy,tuple(pt), 5, (0,0,255), -1)
        cubeImgPts = []
        for i in imgPtsT:
            pt = np.int32(i[0:2]/i[2])
            cubeImgPts.append(tuple(pt))
        for k in range(0,3):
            cv2.line(frameCopy,cubeImgPts[k],cubeImgPts[k+1],(0,0,255),5)
        cv2.line(frameCopy,cubeImgPts[3],cubeImgPts[0],(0,0,255),5)
        for k in range(4,7):
            cv2.line(frameCopy,cubeImgPts[k],cubeImgPts[k+1],(255,0,0),5)
        cv2.line(frameCopy,cubeImgPts[7],cubeImgPts[4],(255,0,0),5)
        for k in range(0,4):
            cv2.line(frameCopy,cubeImgPts[k],cubeImgPts[k+4],(0,255,0),5)
        
        

    
    resized = cv2.resize(frameCopy, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('Cube',resized)
    
    resized = cv2.resize(linaOnImage, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('Lena_img',resized)
    
    resized = cv2.resize(arCodeImage, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('ArCode',resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

cap.release()
cv2.destroyAllWindows()