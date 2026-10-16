import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Define the input arguments with a argparse
#parser = argparse.ArgumentParser(description='Compress a pre-trained model')
#parser.add_argument('--path', type=str, help='path of the image to process')
#parser.add_argument('--mask', type=str, help='1-for circle, 2-eliptical')
#parser.add_argument('--r', type=str, help='radius for the circle mask, eliptical do not apply')
#parser.add_argument('--a', type=str, help='a for the eliptical mask, circle do not apply')
#parser.add_argument('--b', type=str, help='b for the eliptical mask, circle do not apply')
#args=parser.parse_args()




class Fourier():
	'''
		path: path of the image to process
		r: radious for the circular mask
		a: a value for the eliptical mask
		b: b value for the eliptical mask
		mask: used mask for the filter 1: circular, 2: elliptical
	'''
	def __init__(self,
				 path='',
				 mask=1,
				 r=1.0,
				 a=0.0,
				 b=0.0):
		self.path=path
		self.mask=mask
		self.r=r 
		self.a=a 
		self.b=b 
		self.image=[]
		self.f=[]
	
	def _circular_mask(self, rows, cols, r):
		# Find the center
		crow, ccol = int(rows/2), int(cols/2)
		center = [crow, ccol]

		# Create the mask for real and imaginary parts
		mask = np.ones((rows, cols, 2))

		x, y = np.ogrid[:rows, :cols]
		mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
		mask[mask_area] = 0
		return mask


	def _horizontal_eliptical_mask(self, rows, cols, a, b):
		# Find the center
		crow, ccol = int(rows/2), int(cols/2)
		center = [crow, ccol]

		# Create the mask for real and imaginary parts
		mask = np.ones((rows, cols, 2))

		x, y = np.ogrid[:rows, :cols]
		mask_area = ((x - center[0]) ** 2)/a**2 + ((y - center[1]) ** 2)/b**2 <= 1
		mask[mask_area] = 0
		return mask	

	def apply_transformation(self):

		ret,th = cv2.threshold((self.f*255).astype('uint8'), 10, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		u_net_mask=cv2.resize(cv2.imread(self.path.replace('images','masks'), 0), (256,256), interpolation = cv2.INTER_AREA)
		x=np.where(th != 255, self.image, int(np.mean(self.image)))
		x=np.where(u_net_mask != 255, x, self.image)
		return x



	'''
	'''
	def plot_images(self):
		rows=2
		columns=2
		fig=plt.figure(figsize=(10,10))
		fig.add_subplot(rows,columns,1)
		plt.imshow(self.image, cmap='gray')

		fig.add_subplot(rows, columns,2)
		plt.imshow(self.f, cmap='gray')

		#print((self.f*255).astype(int), (self.f*255).astype(int).min(), (self.f*255).astype(int).max())
		ret,th = cv2.threshold((self.f*255).astype('uint8'), 10, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		fig.add_subplot(rows, columns,3)
		plt.imshow(th, cmap='gray')

		#print(self.path.replace('images','masks'))
		#u_net_mask=cv2.resize(cv2.imread(self.path.replace('images','masks'), 0), (256,256), interpolation = cv2.INTER_AREA)
		#x=np.where(th != 255, self.image, int(np.mean(self.image)))
		x=self.apply_transformation()
		fig.add_subplot(rows, columns,4)
		plt.imshow(x, cmap='gray')

		plt.show()


	'''
	'''
	def fourier_transform(self, img):
		# Transformada de fourier
		fourier=cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
		shift=np.fft.fftshift(fourier)
		# get the mask
		_mask=[]
		if self.mask==1:
			_mask = self._circular_mask(img.shape[0], img.shape[1], self.r)
		else:
			_mask = self._horizontal_eliptical_mask(img.shape[0], img.shape[1], self.a, self.b)
		
		# Apply the mask
		masked_shift = shift * _mask
		
		# Inverse Fourier Transform
		f_shift = np.fft.ifftshift(masked_shift)
		filtered_img = cv2.idft(f_shift)
		filtered_img = cv2.magnitude(filtered_img[:, :, 0], filtered_img[:, :, 1])
		# save in the list
		filtered_img=filtered_img/np.max(filtered_img)
		
		#return cv2.resize(filtered_img, (256,256), interpolation = cv2.INTER_AREA)
		return filtered_img


	def set_path(self, path):
		self.image = cv2.resize(cv2.imread(path, 0), (256,256), interpolation = cv2.INTER_AREA)
	'''
	'''
	def run(self, path):
		self.path = path
		self.image = cv2.resize(cv2.imread(self.path, 0), (256,256), interpolation = cv2.INTER_AREA)
		return self.image
		#self.f = self.fourier_transform(self.image)
		#return self.apply_transformation()


#my_fourier_transformer=Fourier(path=args.path,
#							   mask=int(args.mask),
#							   r=float(args.r),
#							   a=float(args.a),
#							   b=float(args.b))
#my_fourier_transformer.run()
