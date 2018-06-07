import numpy as np
import cv2
import time
from sklearn import svm
x = []
y = []

start = time.time()

for i in range (1,11099):
	str1 = '0/'
	str2 = str1 + str(i) + '.png'
	img = cv2.imread(str2)
	img = np.asarray(img)
	img1 = img.flatten('F')
	x.append(img1)
	y.append(0)

for i in range (1,8965):
	str1 = '1/'
	str2 = str1 + str(i) + '.png'
	img = cv2.imread(str2)
	img = np.asarray(img)
	img1 = img.flatten('F')
	x.append(img1)
	y.append(1)

end = time.time()

print(len(x))


clf = svm.SVC(gamma=0.001, C=100)

start1 = time.time()
clf.fit(x,y)
end1 = time.time()

#if(clf.predict(img2)[0]!=i):



white_lbl = cv2.imread('white.png',cv2.IMREAD_GRAYSCALE)
black_lbl = cv2.imread('black.png',cv2.IMREAD_GRAYSCALE)

# print(small_lbl2.shape,lbl[(i*32):((i+1)*32),(j*32):((j+1)*32)].shape)

start2 = time.time()
for k in range(1,6):
	img = cv2.imread('test'+str(k)+'.jpg')
	img = cv2.resize(img,(640,480))
	lbl = img
	lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)
	for i in range(15):
		for j in range(20):
			print(i,j)
			image = img[(i*32):((i+1)*32),(j*32):((j+1)*32)]
			image = image.flatten('F')
			x = clf.predict(image)[0]
			print('preds: ',i,j,x)
			if(x == 0):
				lbl[(i*32):((i+1)*32),(j*32):((j+1)*32)]=black_lbl
			elif(x==1):
				lbl[(i*32):((i+1)*32),(j*32):((j+1)*32)]=white_lbl
	cv2.imwrite('label' + str(k) +'.png',lbl)

end2 = time.time()

print('time:',start-end)
print('time1:',start1-end1)
print('time2:',start2-end2)

cv2.imshow('after',lbl)

