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




def calculate_areascaling(imagefile, plot = False):
    img = cv.imread(imagefile, cv.IMREAD_COLOR)
    height,width=img.shape[:2]
    start_row,start_col=int(height*0.2),int(width*0.2)
    end_row,end_col=int(height*0.8),int(width*0.8)#taglia il bordo
    cropped=img[start_row:end_row,start_col:end_col]
    gray_img = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    img_area = np.shape(gray_img)[0]*np.shape(gray_img)[1]
    ret,thresh = cv.threshold(gray_img,130,255,cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE )
    contours = [c for c in contours if cv.arcLength(c,False)>50]
    if plot:
        cv.drawContours(cropped, contours, -1, (0, 255, 0), 3)
        plt.figure(figsize=(10,10))
        plt.imshow(cropped)
        plt.show()
    tot_area = np.sum([cv.contourArea(cnt) for cnt in contours])
    tot_perimeter = np.sum([cv.arcLength(cnt,True) for cnt in contours])
    scaling = [np.log(cv.contourArea(cnt))/np.log(cv.arcLength(cnt,True)) for cnt in contours]
    mean_scaling = np.average(scaling)
    tot_num = len(contours)
    num_ge_11 = len([s for s in scaling if s >= 1.1])
    return [tot_perimeter/tot_num, tot_area/tot_num, mean_scaling, num_ge_11]

def analyze_images(folder):
    files = [f for f in listdir(folder) if (isfile(join(folder, f))) and f.endswith(".jpg")]
    data_list = []
    for f in files:
        data_list.append(calculate_areascaling(join(folder,f)))
    return [list(x) for x in zip(*data_list)]


def main():
    low_data = analyze_images("./immagini/basse/")
    high_data = analyze_images("./immagini/alte/")
    other_data = analyze_images("./immagini/altro/")

    """plt.figure(figsize=(10,10))
    _= plt.plot(low_data[1],low_data[2], 'o', color="green")
    _= plt.plot(high_data[1],high_data[2], 'o', color="red")
    _= plt.plot(other_data[1],other_data[2], 'o', color="yellow")
    _= plt.xlabel("Total cloud area")
    _= plt.ylabel("Total cloud perimeter")
    plt.show()"""

    Xlow = np.transpose(np.array(low_data))
    ylow = np.ones(np.shape(Xlow)[0])*1
    Xhigh = np.transpose(np.array(high_data))
    yhigh = np.ones(np.shape(Xhigh)[0])*2
    Xother = np.transpose(np.array(other_data))
    yother = np.ones(np.shape(Xother)[0])*3

    Xdata = np.concatenate((Xlow, Xhigh, Xother), axis=0)
    ydata = np.concatenate((ylow, yhigh, yother), axis=0)

    Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, ydata, test_size = 0.30) 

    clf = RandomForestClassifier(n_estimators = 100, max_depth=8)   
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest) 
    joblib.dump(clf, './random_forest.joblib' )
    """print("Model score: ", metrics.accuracy_score(ytest, ypred)) 
    print("Features importance: ",clf.feature_importances_)""" 

if __name__ == "__main__":
    main()