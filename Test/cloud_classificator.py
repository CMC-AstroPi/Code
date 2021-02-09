"""
File for creating the machine learning model.
"""


"""
IMPORT LIBRARIES
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.stats
from os import listdir
from os.path import isfile, join

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import joblib 


"""
FUNCTION CREATION
"""
# function to extract data from the image
def calculate_areascaling(imagefile):
    img = cv.imread(imagefile, cv.IMREAD_COLOR)
    height,width=img.shape[:2]
    start_row,start_col=int(height*0.2),int(width*0.2)
    end_row,end_col=int(height*0.8),int(width*0.8)
    cropped=img[start_row:end_row,start_col:end_col]
    gray_img = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    img_area = np.shape(gray_img)[0]*np.shape(gray_img)[1]
    ret,thresh = cv.threshold(gray_img,130,255,cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE )
    if len(contours)>0:
        contours = [c for c in contours if cv.arcLength(c,False)>50]
    else:
        contours = []
    if len(contours)>0:   
        tot_area = np.sum([cv.contourArea(cnt) for cnt in contours])
        tot_perimeter = np.sum([cv.arcLength(cnt,True) for cnt in contours])
        scaling = [np.log(cv.contourArea(cnt))/np.log(cv.arcLength(cnt,True)) for cnt in contours]
        mean_scaling = np.average(scaling)
        tot_num = len(contours)
        num_ge_11 = len([s for s in scaling if s >= 1.1])
        return [tot_perimeter/tot_num, tot_area/tot_num, mean_scaling, num_ge_11]
    else:
        return [0.0, 0.0 ,0.0 ,0.0]
# -------------------------------------------------------

# image extractions from the folder
def analyze_images(folder):
    files = [f for f in listdir(folder) if (isfile(join(folder, f))) and f.endswith(".jpg")]
    data_list = []
    for f in files:
        data_list.append(calculate_areascaling(join(folder,f)))
    return [list(x) for x in zip(*data_list)]
# -------------------------------------------------------

# main function for machine learning
def main():
    low_data = analyze_images("./immagini/basse/")
    high_data = analyze_images("./immagini/alte/")
    other_data = analyze_images("./immagini/altro/")

    Xlow = np.transpose(np.array(low_data))
    ylow = np.ones(np.shape(Xlow)[0])*1
    Xhigh = np.transpose(np.array(high_data))
    yhigh = np.ones(np.shape(Xhigh)[0])*2
    Xother = np.transpose(np.array(other_data))
    yother = np.ones(np.shape(Xother)[0])*3

    Xdata = np.concatenate((Xlow, Xhigh, Xother), axis=0)
    ydata = np.concatenate((ylow, yhigh, yother), axis=0)

    clf = RandomForestClassifier(n_estimators = 100, max_depth=8)   
    clf.fit(Xdata, ydata) 
    joblib.dump(clf, './random_forest.joblib' )
# -------------------------------------------------------

if __name__ == "__main__":
    main()
