import cv2
import numpy as np
from numpy.lib.type_check import imag
import yaml
import imutils
import math
import torch
 

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#to filter by class ie select only persons
model.classes = [0]
   
# confidence threshold (0-1).
model.conf = 0.5 
 
# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        cv2.circle(img,(x,y),4,(0,255,0),-1)
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])

# Create a black image and a window
windowName = 'MouseCallback'
cv2.namedWindow(windowName)

# Create an empty list of points for the coordinates
list_points = list()

# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)

def getTransform(image,rect):
    (tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth-1, 0],
		[maxWidth-1, maxHeight-1],
		[0, maxHeight-1]], dtype = "float32")
    
    # dst = np.float32([[0, 0],
    #                     [0, maxHeight - 1],
    #                     [maxWidth - 1, maxHeight - 1],
    #                     [maxWidth - 1, 0]])
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, M.astype(np.float32), (480,480),flags=cv2.INTER_LINEAR)
    # warped = cv2.warpPerspective(image, M.astype(np.float32), (maxWidth,maxHeight),flags=cv2.INTER_LINEAR)
	# return the warped image
    return warped,M,maxHeight, maxWidth

def detectHOG(img):
    
    # Initializing the HOG person
    # detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    # Resizing the Image
    img = imutils.resize(img,
                        img.shape[1])
    
    # Detecting all the regions in the 
    # Image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(img, 
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)
    
    imgcopy = np.copy(img)
    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(imgcopy, (x, y), 
                    (x + w, y + h), 
                    (0, 0, 255), 4)

    # Showing the output Image
    # cv2.imshow("Image", imgcopy)
    # cv2.waitKey(1000)
    
    return regions

def detectYOLO(imgBGR):
    imgRGB = imgBGR[:, :, ::-1]

    
    #inference
    results = model(img)
    # results.print()
    results.xyxy[0]
    
    regions = []
    for result in results.xyxy[0]:
            # (x,y,w,h)
        # print(result[0].item())
        # .item() gets value of tensor
        xmin = result[0].item()
        ymin = result[1].item()
        xmax = result[2].item()
        ymax = result[3].item()
        
        x = xmin
        y = ymin
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        
        regions.append((int(x),int(y),int(w),int(h)))
        
    return regions

def transformPoints(bboxes,M,warpedIm):
    birdsEyeHumanCoords = []
    for (x, y, w, h) in bboxes: 

        centre = np.array([[[int(x + w/2), int(y + h/2)]]], dtype= "float32")
        ptArr = cv2.perspectiveTransform(centre, M)
        pt = ptArr[0][0]
        # print("Box centre")
        # print(int(x + w/2), int(y + h/2))
        # print("transformed box centre")
        # print(pt)
        birdsEyeHumanCoords.append(pt)
        

    # warpedImcopy = np.copy(warpedIm)
    # for coord in birdsEyeHumanCoords:
    #     cv2.circle(warpedImcopy,(int(coord[0]),int(coord[1])),10,(255,0,0),-1)        
    # cv2.imshow("warpedImcopy",cv2.resize(warpedImcopy,(int(warpedImcopy.shape[0]/2),int(warpedImcopy.shape[1]/2))))
    # cv2.waitKey(1000)
    
    return birdsEyeHumanCoords

def getPixelDistance(p1,p2,M):
    # p1 and p2 transformed into birds eye view
    # euclidean distance between them is found

    p1arr = np.array([[[int(p1[0]), int(p1[1])]]], dtype= "float32")
    p1 = cv2.perspectiveTransform(p1arr, M)[0][0]

    p2arr = np.array([[[int(p2[0]), int(p2[1])]]], dtype= "float32")
    p2 = cv2.perspectiveTransform(p2arr, M)[0][0]
    
    pd = np.linalg.norm(p1 - p2)
    print("Pixel threshold:")
    print(pd)
    
    return pd

def checkViolations(pts, thresh, warpedIm):

    warpedImcopy = np.copy(warpedIm)
    for pt in pts:
        cv2.circle(warpedImcopy,(int(pt[0]),int(pt[1])),10,(0,255,0),-1) 
        cv2.circle(warpedImcopy,(int(pt[0]),int(pt[1])),50,(0,255,0),5)
        for p in pts:
            
            if not (pt[0] == p[0] and pt[1] == p[1]):
                
                dist = np.linalg.norm(pt - p)
                # print("Distance between " + str(pt) + " and " + str(p) + " is " + str(dist) + " threshold " + str(thresh))
                
                if dist < thresh :
                    
                    #red marks violation
                    cv2.circle(warpedImcopy,(int(pt[0]),int(pt[1])),20,(0,0,255),-1)  
                    cv2.circle(warpedImcopy,(int(p[0]),int(p[1])),20,(0,0,255),-1)  
                    cv2.circle(warpedImcopy,(int(pt[0]),int(pt[1])),50,(0,0,255),5)  
                    cv2.circle(warpedImcopy,(int(p[0]),int(p[1])),50,(0,0,255),5)  
            
    # cv2.imshow("warpedImcopy",warpedImcopy)
    cv2.imshow("warpedImcopy2",cv2.resize(warpedImcopy,(int(warpedImcopy.shape[1]),int(warpedImcopy.shape[0]))))

    cv2.waitKey(1)   # np.linalg.norm(x - y)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi',fourcc, 10.0, (int(warpedImcopy.shape[0]),int(warpedImcopy.shape[1])))
    out.write(warpedImcopy)
  
def getValues(img):
        #extract 6 points, calculate threshold
    while (True):
        cv2.imshow(windowName, img)

        if len(list_points) == 6:

            rect = np.array(list_points[0:4])
                    
            #above four points form the region of interest
            #birds eye view of region of interest is found
            #the transform to birds-eye-view is denoted by M
            warpedIm,M,maxHeight, maxWidth = getTransform(img, rect)
                    
             # points 4 and 5 used to estimate distance of 6ft from the image
            pd = getPixelDistance(list_points[4],list_points[5], M)
            
            break
        if cv2.waitKey(20) == 27:
            break
    
    return M,warpedIm,pd,maxHeight, maxWidth

if __name__ == "__main__":
    
    # first frame of the video used to set Region of Interest
    # and pixel distance constant
    # Click 4 points to define ROI and 2 more points 
    # to define 6ft distance 
    vidcap = cv2.VideoCapture('./vid_short.mp4')
    success,img = vidcap.read()
    # frame_width = int(vidcap.get(3))
    # frame_height = int(vidcap.get(4))
    # size = (frame_width, frame_height)


    #Detect bounding box 
    bboxes = detectYOLO(img)

    #get transform matrix, warped image, pixel distance,maxwidth and height for visualising
    M,warpedIm,pd,maxHeight, maxWidth = getValues(img)
    
    while success:
        
        success,img = vidcap.read()
        #Detect bounding box 
        bboxes = detectYOLO(img) 
        warped = cv2.warpPerspective(img, M.astype(np.float32), (480,480))
       
        # The centre of the bounding boxes of humans
        # is converted to birds eye view using 
        # transform matrix  
        birdsEyePts = transformPoints(bboxes,M,warped)
                
        checkViolations(birdsEyePts, pd, warped)
    
    vidcap.release()
    cv2.destroyAllWindows()