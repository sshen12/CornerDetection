import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

TRANSLATION = 1000000000
DEFUALT = 100000

def preview(name,ig):
	cv2.imshow(name,ig)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	pass

def gsSmooth(n,imgs):
	kernal = np.ones((n,n),np.float32)/(n*n)
	dst = cv2.filter2D(imgs,-1,kernal)
	return dst

def getGradient(img):
	# laplacian64 = cv2.Laplacian(img, cv2.CV_64F)
	sobelx64 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	sobely64 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	# print(laplacian64)

	return sobely64,sobelx64

def HarrisCorner(image, blocksize, k, threshold):
	img = image.copy()
	height = img.shape[0]
	width = img.shape[1]

	# calculate gradient
	dy, dx = getGradient(image)
	Ix = dx**2
	Ixy = dy*dx
	Iy = dy**2

	corner_list = []

	c = int(blocksize/2)

	print("Looking for Corners...")

	for y in range(c, height - c):
	    for x in range(c, width - c):
	    	# take correlation matrix
	    	block_Ix = Ix[ y-c:y+c+1, x-c:x+c+1]
	    	block_Ixy = Ixy[ y-c:y+c+1, x-c:x+c+1]
	    	block_Iy = Iy[ y-c:y+c+1, x-c:x+c+1]
	    	# Compute eigenvalue
	    	eigval1 = np.linalg.eigvals(block_Ix).sum()
	    	eigval2 = np.linalg.eigvals(block_Iy).sum()
	    	eigval = np.linalg.eigvals(block_Ixy).sum()
	    	# plus in harris corner detection
	    	det = eigval1*eigval2 - eigval**2
	    	trace = eigval1 + eigval2
	    	r = det - k*(trace**2)
	    	if r > threshold:
	    		corner_list.append(tuple((x,y,r,det,trace)))

	#sorted by descending order
	return sorted(corner_list, key=lambda n: n[2], reverse = True)

def localization(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)

	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

	res = np.hstack((centroids,corners))
	res = np.int0(res)

	for i in res:
		half_size = int(10/2)
		left_top = (i[0] - half_size, i[1] - half_size)
		right_top = (i[0] + half_size, i[1] + half_size)
		cv2.rectangle(img,left_top,right_top,(0,255,0),1)

		left_top = (i[2] - half_size, i[3] - half_size)
		right_top = (i[2] + half_size, i[3] + half_size)
		cv2.rectangle(img,left_top,right_top,(0,0,255),1)

	return img

def drawCorner(image,corner_list, rectangle_size, thershold, k):
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	for x,y,r,det,trace in corner_list:
		rn = det - k*(trace**2)
		if rn > thershold:
			image.itemset((y, x, 0), 0)
			image.itemset((y, x, 1), 0)
			image.itemset((y, x, 2), 255)
			# draw rectangle
			half_size = int(rectangle_size/2)
			left_top = (x - half_size, y - half_size)
			right_top = (x + half_size, y + half_size)
			cv2.rectangle(image,left_top,right_top,(0,255,0),1)
	return image

def FeatureVector(img1,img2):
	# corner_size = len(corner_list)
	feature_vector_list = []
	# Initiate SIFT detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None) # returns keypoints and descriptors
	kp2, des2 = orb.detectAndCompute(img2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	for i in des1:
		print(i)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	number = 0

	for mat in matches[:50]:

		number += 1
		i = mat.queryIdx
		j = mat.trainIdx

		x1 = kp1[i].pt[0]
		y1 = kp1[i].pt[1]
		x2 = kp2[j].pt[0]
		y2 = kp2[j].pt[1]


		# feature_vector_list.append[[des1[i],des2[j]]]

		cv2.putText(img1,str(number), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
		cv2.putText(img2,str(number), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
	# Draw first 10 matches.
	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10] ,None, flags=2)

	both = np.hstack((img1,img2)) 

	plt.figure(figsize = (10,6))
	plt.axis('off')
	plt.imshow(both)
	plt.show()

	print(feature_vector_list)
	return feature_vector_list

def Mydetection(image,image_, guassian_size, neighborhood_size):
	#create Localization image in local
	localization1 = localization(image.copy())
	cv2.imwrite('img1.jpg',localization1)
	localization2 = localization(image_.copy())
	cv2.imwrite('img2.jpg',localization2)


	win_name = 'My Corner Detection  ' + '  guassian_size = ' + str(guassian_size) + '   neighborhood_size = ' + str(neighborhood_size)

	img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	img_ = cv2.cvtColor(image_,cv2.COLOR_BGR2GRAY)

	img = gsSmooth(guassian_size,img)
	img_ = gsSmooth(guassian_size,img_)
	
	setting = {
		'threshold_default': DEFUALT,
		'k_default': 0.04,
	}

	def thresholdHandler(n):
		setting['threshold_default'] = n*TRANSLATION
		dst = drawCorner(img,corner_list,30,setting['threshold_default'],setting['k_default'])
		dst_ = drawCorner(img_,corner_list_,30,setting['threshold_default'],setting['k_default'])
		both = np.hstack((dst,dst_)) 
		cv2.imshow(win_name,both)

	def kHandler(n):
		setting['k_default'] = n/100
		dst = drawCorner(img,corner_list,30,setting['threshold_default'],setting['k_default'])
		dst_ = drawCorner(img_,corner_list_,30,setting['threshold_default'],setting['k_default'])
		both = np.hstack((dst,dst_)) 
		cv2.imshow(win_name,both)

	gray = np.float32(img)
	gray_ = np.float32(img_)
	corner_list = HarrisCorner(gray, neighborhood_size, setting['k_default'], setting['threshold_default'])
	corner_list_ = HarrisCorner(gray_, neighborhood_size, setting['k_default'], setting['threshold_default'])

	both = np.hstack((img,img_))
	# get feature vector list

	
	threshold = corner_list[0][2]
	threshold_ = corner_list_[0][2]
	if threshold > threshold_: threshold = threshold_

	# create windows
	cv2.namedWindow(win_name)

	# create trackbar
	cv2.createTrackbar('Threshold',win_name,0,int(threshold/TRANSLATION),thresholdHandler)
	cv2.createTrackbar('k value (n/100)',win_name,0,50,kHandler)

	# cv2.createTrackbar('neighborhood size',win_name,0,8,neighborhoodHandler)

	preview(win_name,both)
	# return feature_vector_list 
	return corner_list,corner_list_

def menu(first_img, second_img):
	def help():	
		print('\t\t\tWelcome to corner dection !!!')
		print('Here is the function we have:')
		print(' "h" - HarrisCorner detection -getGradient to compute -localization image and save -with corner box')
		print(' "f" - Feature vector matching with number')
		print(' "q" - quit')

	help()
	while 1:
		option = input("Please select a function('q' to exit):")
		if (option == 'h'):
			guassian_size = int(input("Please input the variance of the guassian (1 to 10):"))
			while not 0 <= guassian_size <= 10 :
				guassian_size = int(input("Please input the variance of the guassian (1 to 10):"))

			neighborhood_size = int(input("Please input neighborhood size (1 to 10):"))
			while not 0 <= neighborhood_size <= 10 :
				neighborhood_size = int(input("Please input neighborhood size (1 to 10):"))

			Mydetection(first_img.copy(), second_img.copy(), guassian_size , neighborhood_size)
		elif(option == 'f'):
			FeatureVector(first_img.copy(),second_img.copy())
		elif(option == 'q'):
			break
		else:
			print("Please input the code again!")
		pass
	
	pass

def main():
	if len(sys.argv) == 3:
		filename1 = sys.argv[1]
		filename2 = sys.argv[2]
		img1 = cv2.imread(filename1)
		img2 = cv2.imread(filename2)
		if img1 is not None:
			menu(img1, img2)
		else:
			print("The image you input", img1 ,"or", img2, "not found!")
	else:
		img1 = cv2.imread('../data/test.jpg')
		img2 = cv2.imread('../data/test1.jpg')
		if img1 is not None:
			menu(img1, img2)
		else:
			print("The image test.jpg and test1.jpg is not found!") 
	pass

if __name__ == '__main__':
	main()
