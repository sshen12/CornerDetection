import cv2
import numpy as np
from scipy import ndimage
import sys
import select

def menu(img):
	option = input("Please select a function('q' to exit):")
	image = img

	def checkEnter():
	    i,o,e = select.select([sys.stdin],[],[],0.0001)
	    for s in i:
	        if s == sys.stdin:
	            input = sys.stdin.readline()
	            return True
	    return False

	def reloadImg():
		print("Reload the image!") 
		ig = IMAGE.copy()
		return ig

	def saveCurrent():
		check = cv2.imwrite('out.jpg',image)
		if check: print("Saved the image!") 
		else: print("fail to save the image!")
		return image

	def imgToGrayscale():
		ig = image.copy()
		try:
			cv2.cvtColor(ig,cv2.COLOR_BGR2GRAY)
		except Exception:
			print("Aready to gray scale!")
			return ig
		ig = cv2.cvtColor(ig,cv2.COLOR_BGR2GRAY)
		print("Converted to gray scale!")
		return ig

	def imgToGrayscale_():
		ig = image.copy()
		try:
			ig[:,:,0]
		except:
			print("The grayscale Image can not be grayscale again")
			return(ig)
		r, g, b = ig[:,:,0], ig[:,:,1], ig[:,:,2]
		grayImg = 0.2989 * r + 0.5870 * g + 0.1140 * b
		print("Converted to gray scale!")
		return grayImg

	def colorChannel():
		print("Start to walk through the color channel! Please Press any key or just put enter to print the color channel.")
		for i in range(0,image.shape[0],1):
			for j in range(0,image.shape[1],1):
				if checkEnter():
					try:
						image[i][j][0]
					except:
						print("The grayscale Image can not be traverse")
						return(image)
					print([image[i][j][0],image[i][j][1],image[i][j][2]])
		print("walk through all the color channel!") 
		return image

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

	def smooth():
		def sliderHandler(n):
			global resultImg
			if n == 0:
				n = 1
			dst = gsSmooth(n,images)
			cv2.imshow(winName,dst)
			resultImg = dst

		winName = 'smoothing track bar'
		images = image
		try:
			images = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
		except:
			print('gray scale done before')

		cv2.namedWindow(winName)
		cv2.createTrackbar('smoothing track bar',winName,0,255,sliderHandler)
		preview(winName,images)

		try:
		  resultImg
		except NameError:
			return images
		else:
			print("Saved the change!") 
			return resultImg

	def smooth_():
		def sliderHandler(n):
			global resultImg
			if n == 0 | 1:
				n = 1.2
			ig = ndimage.gaussian_filter(images,sigma=n)
			cv2.imshow(winName,ig)
			resultImg = ig
		winName = 'Custom smoothing track bar'
		images = image
		try:
			images = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
		except:
			print('gray scale done before')

		cv2.namedWindow(winName)
		cv2.createTrackbar('Custom smoothing track bar',winName,0,30,sliderHandler)
		preview(winName,images)
		try:
			resultImg
		except NameError:
			return ig
		else:
			print("Saved the change!") 
			return resultImg

	def downSample():
		ig = image.copy()
		rows = ig.shape[0]
		cols = ig.shape[1]
		ig = cv2.resize(ig,(round(cols/2),round(rows/2)))
		print("Down sample Done")
		return ig

	def downSampleSmooth():
		ig = image.copy()
		rows = ig.shape[0]
		cols = ig.shape[1]
		dst = gsSmooth(5,ig)
		ig = cv2.resize(ig,(round(cols/2),round(rows/2)))
		print("Down sample with smooth Done")
		return ig
	## main Menu
	def convolutionX():
		ig = imgToGrayscale()
		sobelx = cv2.Sobel(ig,cv2.CV_64F,1,0,ksize=5)
		ig = gsSmooth(5,sobelx) ##convolution here
		cv2.normalize(ig,ig,0,255,cv2.NORM_MINMAX)
		print("aplied sobel x and normalized")
		print(ig)
		return ig

	def convolutionY():
		ig = imgToGrayscale()
		sobely = cv2.Sobel(ig,cv2.CV_64F,0,1,ksize=5)
		ig = gsSmooth(5,sobely)
		cv2.normalize(ig,ig,0,255,cv2.NORM_MINMAX)
		print("aplied sobel y and normalized")
		print(ig)
		return ig

	def magnitude():
		ig = image.copy()
		dx = cv2.Sobel(ig,cv2.CV_64F,1,0,ksize=5)
		dy = cv2.Sobel(ig,cv2.CV_64F,0,1,ksize=5)
		dx = gsSmooth(5,dx)
		dy = gsSmooth(5,dy)
		re = np.sqrt((dx**2) + (dy**2))
		cv2.normalize(re,re,0,255,cv2.NORM_MINMAX)
		print('normalized:',re)
		# dxAbs = cv2.convertScaleAbs(dx)
		# dyAbs = cv2.convertScaleAbs(dy)
		# mag = cv2.addWeighted(dxAbs, 0.5, dyAbs, 0.5, 0)
		# print('weighted normalize to [0,255]:',mag)
		return re

	def plotImage():
		ig = imgToGrayscale()
		def sliderHandler(n):
			global resultImg
			ig = image.copy()
			if n == 0:
				n = 1
			for i in range(0,image.shape[0],n):
				for j in range(0,image.shape[1],n):
					if int(dx[i][j]) != 0 | int(dy[i][j]) != 0:
						k = 10/np.sqrt(dx[i][j]**2 + dy[i][j]**2)
						pt1 = (j,i)
						pt2 = (j + int(dy[i][j]*k),i + int(dx[i][j]*k))
						cv2.arrowedLine(ig, pt1, pt2, (60,20,220), 1)
			cv2.imshow(winName,ig)
			resultImg = ig
		rows = ig.shape[0]
		cols = ig.shape[1]
		# X,Y=np.meshgrid(np.arange(0,cols),np.arange(0,rows))
		winName = 'vectors track bar'
		dx = cv2.Sobel(ig,cv2.CV_64F,1,0,ksize=5)
		dy = cv2.Sobel(ig,cv2.CV_64F,0,1,ksize=5)
		images = image.copy()

		cv2.namedWindow(winName)
		cv2.createTrackbar('vectors track bar',winName,0,int(cols/2),sliderHandler)
		preview(winName,images)
		# vectors = np.array(vectors)
		# print(vectors)
		# X,Y = zip(*vectors)
		# plt.figure()
		# ax = plt.gca()
		# ax.quiver(,dx, dy, angles='xy', scale_units='xy', scale=1)
		# ax.set_xlim([-255, 255])
		# ax.set_ylim([-255, 255])
		# plt.show()
		# plt.draw()
		# plt.pause(0.001)
		return resultImg

	def rotateImage():
		def sliderHandler(n):
			global resultImg
			angle = n
			M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
			dst = cv2.warpAffine(images,M,(cols,rows))
			cv2.imshow(winName,dst)
			resultImg = dst

		winName = 'Rotation track bar'
		images = image
		rows = images.shape[0]
		cols = images.shape[1]
		try:
			images = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
		except:
			print('gray scale done before')

		cv2.namedWindow(winName)
		cv2.createTrackbar('Rotation track bar',winName,0,360,sliderHandler)
		preview(winName,images)
		# points
		# points = np.array(images)
		# # add ones
		# ones = np.ones(shape=(len(points), 1))
		# points_ones = np.hstack([points, ones])
		# # transform points
		# re = resultImg.dot(points_ones.T).T
		return resultImg

	Menu = {
	  'i': reloadImg,
	  'w': saveCurrent,
	  'g': imgToGrayscale,
	  'G': imgToGrayscale_,
	  'c': colorChannel,
	  's': smooth,
	  'S': smooth_,
	  'd': downSample,
	  'D': downSampleSmooth,
	  'x': convolutionX,
	  'y': convolutionY,
	  'm': magnitude,
	  'p': plotImage,
	  'r': rotateImage,
	}
	if option in Menu.keys():
		image = Menu[option]()
		menu(image)
	elif option != 'q':
		print("your input key not found!")
		menu(image)


def main():
	global IMAGE
	def captureImage():
		retval, image = cap.read()
		return image

	if len(sys.argv) == 2:
		filename = sys.argv[1]
		img = cv2.imread(filename)
		if img is not None: 
			print(img.shape)
			IMAGE = img.copy()
			menu(img)
		else:
			print("The image '", filename, "not found!")
	elif len(sys.argv) < 2:
		cap = cv2.VideoCapture(0)
		img = captureImage()
		for i in range(15):
			temp = captureImage()
		img = captureImage()
		check = cv2.imwrite('myImage.jpg',img)
		cap.release()
		while check:
			IMAGE = img.copy()
			menu(img)
			break
	pass

IMAGE = ""

if __name__ == '__main__':
	main()
