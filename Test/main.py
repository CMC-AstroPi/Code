"""
File for capturing and detecting images and the earth's magnetic field.

Link of github: https://github.com/CMC-AstroPi
"""


"""
INCLUSION OF LIBRARIES
"""
# I import this library to know the date and time
import datetime
# -------------------------------------------------------

# I import this library to set the paths
from pathlib import Path
# -------------------------------------------------------

# I import this library to put the delays
from time import sleep
# -------------------------------------------------------

# I import this library to allow me to obtain the coordinates of the ISS
import reverse_geocoder as rg
from ephem import readtle, degree
# -------------------------------------------------------

# I import this library to allow me to detect the magnetic field value with the magnetometer
from sense_hat import SenseHat
# -------------------------------------------------------

# I import these libraries to take photos
from picamera import PiCamera
from picamera.array import PiRGBArray
# -------------------------------------------------------

# I import these libraries to log
import logging
import logzero
# -------------------------------------------------------

# I import this library cv
import cv2 as cv
# -------------------------------------------------------

# I import this library to do operations
import numpy as np
# -------------------------------------------------------

# I import this library for making graphics
import matplotlib.pyplot as plt
# -------------------------------------------------------

# I import these libraries for machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics 
import scipy.stats
import joblib
# -------------------------------------------------------

# I import these libraries for operating system operations
import os
from os import listdir
from os.path import isfile, join
# -------------------------------------------------------

# I import this library to do the magnetic field strength calculations
import math
# -------------------------------------------------------


"""
SET VARIABLE
"""
# parameter setting for obtaining latitude and longitude
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   20316.41516162  .00001589  00000+0  36499-4 0  9995"
line2 = "2 25544  51.6454 339.9628 0001882  94.8340 265.2864 15.49409479254842"

iss = readtle(name, line1, line2)
# -------------------------------------------------------

# parameter setting for time measurement
INITIAL_TIME = datetime.datetime.now()
# -------------------------------------------------------

# parameter setting for magnetic field detection
sensor = SenseHat()
# -------------------------------------------------------

# setting parameters to take a photo
numberPhoto = 0
camera = PiCamera()
camera.resolution = (2592, 1944)
TIME_BETWEEN_TWO_SHOTS = 25
# -------------------------------------------------------


"""
SET FILES
"""
# opening and creating data file
file_data = logzero.setup_logger(name='file_data', logfile='./file_data.csv')
file_data.info(',ID_Photo,PhotoType,MagnetometerValue,Date_Time,Timer,Latitude,Longitude,Position')
# -------------------------------------------------------

# opening info error file 
file_info_error= logzero.setup_logger(name='file_info_error', logfile='./file_info_error.csv')
# -------------------------------------------------------

# opening info time file 
file_info_time= logzero.setup_logger(name='file_info_time', logfile='./file_info_time.csv')
file_info_time.info(',ID_Photo,PhotoCaptureRunTime,IsDayRunTime,CalculateAreascalingRunTime,MachineLearningRuntime')
# -------------------------------------------------------

# file opening for machine learning
loaded_rf = joblib.load("random_forest.joblib")
# -------------------------------------------------------


"""
CREATE BASIC FUNCTION
"""
# function to get latitude
def getLatitude():
    iss.compute()
    return(iss.sublat/degree)
# -------------------------------------------------------

# function to get longitude
def getLongitude():
    iss.compute()
    return(iss.sublong/degree)
# -------------------------------------------------------

# function to get the name of the city where the ISS is located
def getPosition():
    string = ""
    pos = (getLatitude(), getLongitude())
    location = rg.search(pos)
    return "lat: "+str(location[0]["lat"])+"-"+"lon: "+str(location[0]["lon"])+"-"+"city-name: "+str(location[0]["name"])
# -------------------------------------------------------

# function to get date and time
def getHourAndDate():
    return(datetime.datetime.now())
# -------------------------------------------------------

# function to get the timer value
def getTimer():
    return (getHourAndDate()-INITIAL_TIME)
# -------------------------------------------------------

# function to know the number of the photo taken
def getNumerPhoto():
    global numberPhoto
    numberPhoto+=1
    return numberPhoto
# -------------------------------------------------------

# function to obtain the values ​​recorded by the magnetometer
def getMagnetometricSensor():
    magnetometer_values = sensor.get_compass_raw()
    mag_x, mag_y, mag_z = magnetometer_values['x'], magnetometer_values['y'], magnetometer_values['z']
    return str(math.sqrt(math.pow(mag_x,2)+math.pow(mag_y,2)+math.pow(mag_z,2)))
# -------------------------------------------------------

# function to create the name of the photo
def createNamePhoto():
    return('-img_'+str(getNumerPhoto())+'.png')
# -------------------------------------------------------

# function to convert images into data for machine learning
def calculate_areascaling(imagefile, plot = False):
    img = cv.imread(imagefile, cv.IMREAD_COLOR)
    height,width=img.shape[:2]
    start_row,start_col=int(height*0.2),int(width*0.2)
    end_row,end_col=int(height*0.8),int(width*0.8)
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

# image capture function 
def makePhoto():
    times = []
    nameUltimatePhoto = createNamePhoto()

    # photo capture
    try:
        initial_time = getTimer()
        camera.capture(nameUltimatePhoto)
        times.append(getTimer() - initial_time)
    except Exception as e:
        file_info_error.error("photo capture error")

    # Convert the color from RGB to Grayscale
    img = cv.imread(nameUltimatePhoto)
    try:
        initial_time = getTimer()
        v_day = is_day(img)
        times.append(getTimer() - initial_time)
    except Exception as e:
            file_info_error.error("is_day error")

    if(v_day):

        # machine learning
        try:
            initial_time = getTimer()
            data_photo = calculate_areascaling(nameUltimatePhoto)
            times.append(getTimer() - initial_time)

            initial_time = getTimer()
            value = machineLearning(data_photo)[0]
            times.append(getTimer() - initial_time)
            

            if(value == 1.):
                new_name = 'low_clouds'
            elif(value==2.):
                new_name = 'high_clouds'
            else:
                new_name = 'other'
            os.rename(nameUltimatePhoto, new_name+nameUltimatePhoto)
            saveData(new_name)
        except Exception as e:
            file_info_error.error("machine learning error")
    else:
        os.remove(nameUltimatePhoto)
        global numberPhoto
        numberPhoto -= 1
    file_info_time.info(", %d, %s", numberPhoto, str(times))

    sleep(TIME_BETWEEN_TWO_SHOTS)
# -------------------------------------------------------

# function to save data in the csv file
def saveData(photo_type):
    file_data.info(', %d, %s, %s, %s, %s, %f, %f, %s', numberPhoto, photo_type, getMagnetometricSensor(), str(getHourAndDate()), str(getTimer()), getLatitude(), getLongitude(), str(getPosition()))
# -------------------------------------------------------

# function to understand if the image contains low clouds
def machineLearning(img):
    lista = []
    lista.append(img)
    result = loaded_rf.predict(lista)
    print(result)
    return result
# -------------------------------------------------------

# function to understand if it is dark or not
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

# main function
def run():
    while(getHourAndDate() < INITIAL_TIME + datetime.timedelta(minutes=178)):
        makePhoto()
            
# -------------------------------------------------------

run()
