Readme

basicProcessing.py:

Menu
'i' - Reload the original image
'w' - Save the current image into file out.jpg
'g' - Convert to grayscale using openCV
'G' - Convert to grayscale using my function
'c' - Cycle through the color Channel(press key and enter to print the color channel)
's' - Convert to grayscale and smooth using openCV with track bar
'S' - Convert to grayscale and smooth using my function with track bar
'd' - Down sample without smooth
'D' - Down sample with smooth
'x' - perform convolution with an X derivative after grayscale conversion
'y' - perform convolution with an Y derivative after grayscale conversion
'm' - Show the magnitude of the gradient normalized to the range [0,255]
'p' - Plot the gradient vectors of image every N pixels in length k after grayscale conversion
'r' - Rotate by some angle after grayscale conversion





Corner Detection and Matching.py

Load and display two images containing similar content. In each image perform: 
- Image gradient and apply Harris corner detection algorithm 
- Obtain localization of each corner 
- Compute a feature vector for each corner point 
- Display the corners by drawing rectangles.

Using the feature vectors computed to match and number the feature point.

Interactively controlled parameters: The variance of the Gaussian, the neighborhood size for computing the correlation matrix, the weight of the trace and a threshold value.
