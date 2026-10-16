from Fourier_preprocess import Fourier
import argparse
import cv2
import numpy as np
from utils import readFolder
from architectures import VGG11, alexNet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from xplique.attributions import GradCAM
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Define the input arguments with a argparse
parser = argparse.ArgumentParser(description='Make an experiment with Fourier transform')
parser.add_argument('--dataset', type=str, help='path of the image to process')
parser.add_argument('--lr', type=str, help='Learning rate for the adam optimizer')
parser.add_argument('--beta_1', type=str, help='beta 1 for the adam optimizer')
parser.add_argument('--beta_2', type=str, help='beta 2 rate for the adam optimizer')
parser.add_argument('--eps', type=str, help='epsilon rate for the adam optimizer')
parser.add_argument('--mask', type=str, help='1-for circle, 2-eliptical')
parser.add_argument('--r', type=str, help='radius for the circle mask, eliptical do not apply')
parser.add_argument('--exp_id', type=str, help='exeriment idetifier to save the model')
args=parser.parse_args()



class Experiment():


	def __init__(self,
		base_folder='Kaggle_v3',
		lr=0.0001,
		beta_1=0.99,
		beta_2=0.99,
		epsilon=1e-7,
		r=1.0,
		exp_id='1'):
		'''
		Initialize the experiment object

		Args: 
			lr: learning rate for the Adam Optimizer
			beta_1: beta_1 for the Adam Optimizer
			beta_2: beta_2 for the Adam Optimizer
			epsilon: epsilon for the Adam Optimizer
			base_folder: folder where the dataset is located
			folders: folders where the classes are located (inside base_folder)
			pre_processor: Fourier preprocessor to pre-process the images
			r: radious of the circular mask of the pre-processer
			X: list where the dataset will be append
			y: objective function of the dataset (classes)
			masks: list where the masks associated to X
			model: model used for train the agent
			exp_id: identifier to save the model
		'''
		self.lr=lr 
		self.beta_1=beta_1
		self.beta_2=beta_2
		self.epsilon=epsilon
		self.base_folder=base_folder
		self.folders=['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
		self.pre_processer=Fourier(r=r)
		self.r=r
		self.X=[]
		self.Y=[]
		self.masks=[]
		#self.model=VGG11(len(self.folders))
		self.model=alexNet(len(self.folders))
		self.exp_id=exp_id

	def read_data(self):
		
		# Open the dataset images to analyze
		x=0
		for folder in self.folders:
			for img_p in readFolder(self.base_folder+'/'+folder+'/images/'):
				# open the image
				self.X.append(self.pre_processer.run(self.base_folder+folder+'/images/'+img_p))
				# open the masks
				self.masks.append(np.where(cv2.resize(cv2.imread(self.base_folder+folder+'/masks/'+img_p, 0), (256,256), interpolation = cv2.INTER_AREA), 1.0 ,0.0))
				self.Y.append(x)
			x+=1


	def splits(self):
		'''
			Get the partitions of the dataset
		'''
		# stratify according to the classes
		# Split the three arrays
		# 1) 80% for train and 20% to test
		raw_train, raw_test, y_train, y_test=train_test_split(np.rollaxis(np.array([self.X, self.masks]),1,0), self.Y, test_size=0.20, stratify=self.Y)
		raw_train=np.rollaxis(raw_train,1,0)[0]
		self.raw_test, self.test_masks=np.rollaxis(raw_test,1,0)
		# 2) from 80% to train -> 80% train, 20% val
		raw_train, raw_val, y_train, y_val =train_test_split(raw_train, y_train, test_size=0.20, stratify=y_train)
		# convert all the data to numpy arrays
		self.x_train=np.rollaxis(np.array([np.array(raw_train)]),0, 4)
		self.x_test=np.rollaxis(np.array([np.array(np.rollaxis(raw_test,1,0)[0])]), 0, 4)
		self.x_val=np.rollaxis(np.array([np.array(raw_val)]), 0, 4)
		self.y_train=np.array(y_train)
		self.y_test=np.array(y_test)
		self.y_val=np.array(y_val)
		


	def train(self):
		'''
			Train the model and save the model and statistics
		'''
		# Callback to sace the best model according to the validation accuracy
		name=self.exp_id+'_best_model.h5'
		save_best_cb=ModelCheckpoint(filepath=name, monitor='val_accuracy', save_best_only=True)

		
		# Parameters for the optimizer
		opt=Adam(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
		# Compile and train the model
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		# Train the model
		history=self.model.fit(self.x_train, 
			              to_categorical(self.y_train), 
			              validation_data=(self.x_val, to_categorical(self.y_val)),
			              batch_size=32,
			              epochs=20,
			              callbacks=[save_best_cb])

		#load the weights of the best model
		self.model.load_weights(name)
		# evaluate the model with the testing data
		y_pred=np.argmax(self.model.predict(self.x_test), axis=1)
		self.accuracy=accuracy_score(self.y_test, y_pred)

	def LoT(self):
		'''
			We evaluate the intepretability via LoT
			     sum(gradcam (hadamard) mask)
			Lot=---------------------------
			           sum(gradcam)
		'''
		# Get the gradcams of the testing set
		explainer = GradCAM(self.model)
		explanaitions = explainer(self.x_test, to_categorical(self.y_test))
		Lot_g=[]
		for gradcam, mask in zip(explanaitions, self.test_masks):
			# If we do not obtain values in gradcam we return zero
			if np.sum(gradcam) == 0.0:
				Lot_g.append(0.0)
			# else return the Lot evaluation function
			else:
				Lot_g.append(np.sum(np.multiply(gradcam, mask))/np.sum(gradcam))
		# the interpretability will be the mean of the LoT
		self.interpretability=np.mean(np.array(Lot_g))

	def gradcams(self):
		# Get the gradcams of the testing set
		explainer = GradCAM(self.model)
		explanaitions = explainer(self.x_test, to_categorical(self.y_test))
		# plot the attribution on top of the image
		for i in range(10):
			#plt.subplot(1, len(X), i+1)
			#plt.imshow(X[i]/255.0)
			plt.axis('off')
			plt.imshow(self.x_test[i]/255.0)
			plt.imshow(explanaitions[i], cmap="jet", alpha=0.3)
			plt.savefig('explanaitions_VGG_11/'+str(i)+'_explanations_.jpg')
		

		# Get the gradcams of the testing set
		explainer = GradCAM(self.model)
		explanaitions = explainer(self.raw_test, to_categorical(self.y_test))
		# plot the attribution on top of the image
		for i in range(10):

			#plt.subplot(1, len(X), i+1)
			#plt.imshow(X[i]/255.0)
			plt.axis('off')
			plt.imshow(self.raw_test[i]/255.0)
			plt.imshow(explanaitions[i], cmap="jet", alpha=0.3)
			plt.savefig('explanaitions_VGG_11/'+str(i)+'_explanations.jpg')


	def print_results(self):
		'''
			print results in the log file and in the terminal
		'''
		line = str(self.lr)+','+str(self.beta_1)+','+str(self.beta_2)+','+str(self.epsilon)+','+str(self.r)+','+str(self.accuracy)+','+str(self.interpretability)+','+str(self.accuracy*0.4+self.interpretability*0.6)+'\n'
		f = open('population.log','a')
		f.write(line)
		f.close()
		print(self.lr, ',', self.beta_1, ',', self.beta_2, ',', self.epsilon, ',', self.r, ',', self.accuracy, ',', self.interpretability, ',', self.accuracy*0.4+self.interpretability*0.6)
	
	def evaluate(self, model):
		self.model=load_model(model)
		self.read_data()
		self.splits()
		self.gradcams()

	def run(self):
		self.read_data()
		self.splits()
		self.train()
		self.LoT()
		self.print_results()



exp=Experiment(base_folder=args.dataset,
			   lr=float(args.lr),
			   beta_1=float(args.beta_1),
			   beta_2=float(args.beta_2),
			   epsilon=float(args.eps),
			   r=float(args.r),
			   exp_id=str(args.exp_id))
exp.run()
#exp=Experiment()
#exp.evaluate('VGG11_Variabilidad/6_best_model.h5')
