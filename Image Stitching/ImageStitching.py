import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
MIN_MATCH_COUNT = 10
UBIT = 'shrishti'
np.random.seed(sum([ord(c) for c in UBIT]))

def task11(im,imName):
    image = cv2.imread(im)
    #grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #apply sift
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(image, None)
    
    img=cv2.drawKeypoints(gray,kps,image)
    cv2.imwrite(imName,img)

task11('mountain1.jpg','task1_sift1.jpg')
task11('mountain2.jpg','task1_sift2.jpg')
img1 = cv2.imread('mountain1.jpg') # queryImage
img2 = cv2.imread('mountain2.jpg') # trainImage
 
gray1 = cv2.imread('mountain1.jpg',0) # queryImage
gray2 = cv2.imread('mountain2.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
    
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
i=0
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        #good.append([m])
        good.append(m)
        

img3 = cv2.drawMatches(gray1,kp1,gray2,kp2,good,None,flags=2)
cv2.imwrite('task1_matches_knn.jpg',img3)
#plt.imshow(img3),plt.show()

#random point selection
good10=random.sample(good,len(good))

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    print("The Homography Matrix is:")
    print(H)
    #print(mask)
    matchesMask = mask.ravel().tolist()
    ran_pts1=[]
    ran_pts2=[]
    #print(matchesMask)
    for i in [np.random.randint(0,len(good) -1 ) for x in range(10)]:
        ran_pts1.append(good[i])
        ran_pts2.append(matchesMask[i])

    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
else:
    print("no match")
    matchesMask = None
    

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = ran_pts2, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,ran_pts1,None,**draw_params)

cv2.imwrite("task1_matches.jpg",img3)
rows1, cols1 = img2.shape[:2]
rows2, cols2 = img1.shape[:2]

list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

translation_dist = [-x_min, -y_min]
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img2
cv2.imwrite("panorama.jpg",output_img)
#plt.imshow(output_img, 'gray'),plt.show()

