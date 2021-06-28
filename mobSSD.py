import torch
import cv2
import numpy as np

# img_path = "trialimg.png"
# #BGR to RGB
# img = cv2.imread(img_path)[:, :, ::-1]

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.65  # confidence threshold (0-1)
model.classes = [0]
vidcap = cv2.VideoCapture('./vid_short.mp4')
success,img = vidcap.read()

frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
   
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, size)
   
# cv2.imshow("Image", img)
# cv2.waitKey(100)
while success:

    success,img = vidcap.read()
    
    imgcopy = np.copy(img)

    
    imgRGB = img[:, :, ::-1]
    results = model(imgRGB)
    # results.print()
    # print(results.xyxy[0])

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
        w = xmax - xmin
        h = ymax - ymin
    
        
        regions.append((int(x),int(y),int(w),int(h)))
        # regions.append(result.xyxy[0])
            
    for (x, y, w, h) in regions:
        # print((x,y,w,h))
        cv2.rectangle(imgcopy, (x, y), 
                        (x + w, y + h), 
                        (0, 0, 255), 4)
        
    cv2.imshow("Image", imgcopy)
    cv2.waitKey(1)
    
    out.write(imgcopy)
  
vidcap.release()
cv2.destroyAllWindows()