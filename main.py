"""
INCLUSION OF LIBRARIES
"""
# importo questa libreria per sapere data e ora
import datetime
# -------------------------------------------------------

# inmporto questa libreria per settare i percorsi
from pathlib import Path
# -------------------------------------------------------

# importo questa libreria per mettere i delay
from time import sleep
# -------------------------------------------------------

# importo queste librerie per permettermi date le coordinate di ottenere il nome della citta con quelle coordinate
import reverse_geocoder as rg
from ephem import readtle, degree
# -------------------------------------------------------

# importo questa libreria per permettermi di ottenere le coordinate della ISS
from ephem import readtle, degree
# -------------------------------------------------------

# importo questa libreria per permettermi di rilevare il valore del campo magnetico con il magnetometro
from sense_hat import SenseHat
# -------------------------------------------------------

# importo questa libreria per fare le foto
from picamera import PiCamera
from picamera.array import PiRGBArray
# -------------------------------------------------------

# importo queste librerie per fare i log
import logging
import logzero
# -------------------------------------------------------

# importo la libreria os
import os
# -------------------------------------------------------

#importo la variabile cv
import cv2 as cv
# -------------------------------------------------------


# importo questa libreria per fare operrazioni
import numpy as np
# -------------------------------------------------------

# importo questa libreria per fare i grafici
import matplotlib.pyplot as plt
# -------------------------------------------------------

# importo per ...
import scipy.stats
# -------------------------------------------------------

# importo queste librerie per il machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics  
# -------------------------------------------------------

# importo questa libreria per operazioni sul sistema operativo
import os
from os import listdir
from os.path import isfile, join
# -------------------------------------------------------

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.ensemble import RandomForestClassifier

import math

"""
SET VARIABLE
"""

# settaggio parametri per ottenimento latitudine e longitudine
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   20316.41516162  .00001589  00000+0  36499-4 0  9995"
line2 = "2 25544  51.6454 339.9628 0001882  94.8340 265.2864 15.49409479254842"

iss = readtle(name, line1, line2)
# -------------------------------------------------------

# settaggio parametri per misurazione tempo
INITIAL_TIME = datetime.datetime.now()
# -------------------------------------------------------

# settaggio parametri per rilevazione campo magnetico
sensor = SenseHat()
# -------------------------------------------------------

# settaggio parametri per fare una foto
numberPhoto = 0
camera = PiCamera()
camera.resolution = (2592, 1944)
TIME_BETWEEN_TWO_SHOTS = 60
# -------------------------------------------------------

loaded_rf = joblib.load("random_forest.joblib")

"""
SET FILES
"""


# apertura e creazione celle file dati
file_data = logzero.setup_logger(name='file_data', logfile='./file_data.csv')
file_data.info(',ID_Photo,MagnetometerValue,Date_Time,Timer,Latitude,Longitude,Position')
# -------------------------------------------------------

# apertura file info 
file_info= logzero.setup_logger(name='file_info', logfile='./file_info.csv')
# -------------------------------------------------------


"""
CREATE BASIC FUNCTION
"""
# funzione per ottenere la latitudine
def getLatitude():
    iss.compute()
    return(iss.sublat/degree)
# -------------------------------------------------------

# funzione per ottenere la longitudine
def getLongitude():
    iss.compute()
    return(iss.sublong/degree)
# -------------------------------------------------------

# funzione per ottenere il nome della città su cui si trova la ISS
def getPosition():
    string = ""
    pos = (getLatitude(), getLongitude())
    location = rg.search(pos)
    str_location = str(location).split(',')
    for i in str_location:
        string += str(i)
    return string
# -------------------------------------------------------

# funzione per ottenere data e ora
def getHourAndDate():
    return(datetime.datetime.now())
# -------------------------------------------------------

# funzione per ottenere il valore del timer
def getTimer():
    return (getHourAndDate()-INITIAL_TIME)
# -------------------------------------------------------

# funzione per sapere il numero della foto fatta
def getNumerPhoto():
    global numberPhoto
    numberPhoto+=1
    return numberPhoto
# -------------------------------------------------------

# funzione per ottenere i valori registrati dal magnetometro
def getMagnetometricSensor():
    magnetometer_values = sensor.get_compass_raw()
    mag_x, mag_y, mag_z = magnetometer_values['x'], magnetometer_values['y'], magnetometer_values['z']
    return str(math.sqrt(math.pow(mag_x,2)+math.pow(mag_y,2)+math.pow(mag_z,2)))
# -------------------------------------------------------

# funzione per creare il nome del foto
def createNamePhoto():
    return('img_'+str(getNumerPhoto())+'.png')
# -------------------------------------------------------

# funzione per convertire le immagini in dati per il machine learning
def calculate_areascaling(imagefile, plot = False):
    img = cv.imread(imagefile, cv.IMREAD_COLOR)
    height,width=img.shape[:2]
    start_row,start_col=int(height*0.2),int(width*0.2)
    end_row,end_col=int(height*0.8),int(width*0.8)#taglia il bordo
    cropped=img[start_row:end_row,start_col:end_col]
    gray_img = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    img_area = np.shape(gray_img)[0]*np.shape(gray_img)[1]
    ret,thresh = cv.threshold(gray_img,130,255,cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE )
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
# -------------------------------------------------------

# funzione per acquisizione immagine 
def makePhoto():
    nameUltimatePhoto = createNamePhoto()  # l'ho salvato dentro una variabile in modo che se l'immagine non andasse bene potesse essere eliminta facilmente
    camera.capture(nameUltimatePhoto)

    # Convert the color from RGB to Grayscale
    img = cv.imread(nameUltimatePhoto)

    if(is_day(img)):
        value = machineLearning(calculate_areascaling(nameUltimatePhoto))
        print(value)
        if(value == 1):
            new_name = 'low_clouds'
        elif(value==2):
            new_name = 'high_clouds'
        else:
            new_name = 'other'
        os.rename(nameUltimatePhoto, new_name+nameUltimatePhoto)
        saveData()
    else:
        os.remove(nameUltimatePhoto)
        global numberPhoto
        numberPhoto -= 1


    sleep(TIME_BETWEEN_TWO_SHOTS)
# -------------------------------------------------------

# funzione per salvare i dati nel file csv
def saveData():
    file_data.info(', %d, %s, %s, %s, %f, %f, %s', numberPhoto, getMagnetometricSensor(), str(getHourAndDate()), str(getTimer()), getLatitude(), getLongitude(), str(getPosition()))
# -------------------------------------------------------

# funzione per capire se l'immagine contiene le nuvole basse
def machineLearning(img):
    result = loaded_rf.predict(img)
    print(result)
    return result
# -------------------------------------------------------

# funzione per capire se e' buio o no
def is_day(img, size_percentage=30, min_threshold=80):
    '''
    Function that return true if in the center size percentage of the photo
    (converted to gray color scale) the average color value is more bright 
    than min_threshold (so, more simply, if it's day).
    '''

    # Get image size
    height, width, _ = img.shape

    # Calculate center coordinate
    centerX = (width // 2 )
    centerY = (height // 2)

    # Calculate RightBorder 
    XRB = centerX + ((width * size_percentage) // 200)                    
    # Calculate LeftBorder
    XLB = centerX - ((width * size_percentage) // 200)
    # Calculate TopBorder
    YTB = centerY + ((height * size_percentage) // 200)
    # Calculate BottomBorder
    YBB = centerY - ((height * size_percentage) // 200)

    bgr_list = []

    # Creation of a list of BGR values for every pixel
    for x in range(XLB,XRB):
        for y in range(YBB,YTB):
            bgr_list.append(img[y,x]) # Append the BGR value to the list

    # Convert bgr_list in a numpy array
    numpy_bgr_array = np.array(bgr_list)
    # Calculate the average value of blue, green and red
    average_value = np.average(numpy_bgr_array,axis=0)

    # Convert the type of datas
    average_value = average_value.astype(int)

    # Map values in uint8 format type
    average_value = np.uint8([[[average_value[0],average_value[1],average_value[2]]]]) 

    # Convert the color from BGR to Grayscale
    gray_avg_value = cv.cvtColor(average_value, cv.COLOR_BGR2GRAY)
    #remove single-dimensional entries from the shape of the numpy array
    gray_avg_value = np.squeeze(gray_avg_value)
    print(gray_avg_value >= min_threshold)
    # Return True if the gray_avg value
    return gray_avg_value >= min_threshold
# -------------------------------------------------------

# funzione principale
def run():
    while(getHourAndDate() < INITIAL_TIME + datetime.timedelta(minutes=178)):
        try:
            makePhoto()
        except Exception as e:
            file_info.error("photo capture error")
# -------------------------------------------------------

run()
#DA ONSERIRE LA FORMULA PER IL CAMPO MAGNETICO E IDMUNUIRE IL RETTANGOLO DI ANALISI 