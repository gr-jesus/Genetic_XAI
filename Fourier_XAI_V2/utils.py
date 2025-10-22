from os import walk
import cv2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc 


#-----------------------------------------------------------------------------------------------------------
# Some additional functions 
#-----------------------------------------------------------------------------------------------------------
def readFolder(mypath):
  '''
  Read the files from a folder

  params:
    mypath: path from we want to read the file name

  return: a list with the files name in the folder
  '''
  return next(walk(mypath), (None, None, []))[2]

def read_image(name):
  '''
  Read and preprocess a image

  params:
    name: path of the image to pre-process

  return:
    pre-processed image for give to the model
  '''
  img = image.cv2.imread(name)
  #x = image.img_to_array(img)
  #x = np.expand_dims(x, axis=0)
  return preprocess_input(x)

def print_clasification_report(y_pred, y_test, target_names):
  '''
  Print the classification report, accuracy, recall, and the confusion matrix

  params:
    y_pred: prediction of the trained model
    y_test: true labels to evaluate the model
  '''
  print('Accuracy:', accuracy_score(y_test, y_pred))
  print('------------------Classification Report------------------')
  print(classification_report(y_test, y_pred, target_names=target_names))
  print('------------------Confusion Matrix------------------')
  print(confusion_matrix(y_test, y_pred))
  print('------------------AUC------------------')
  fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)
  print('Area Under ROC curve',auc(fpr, tpr))