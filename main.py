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
    pos = (getLatitude(), getLongitude())
    location = rg.search(pos)
    return location
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
    return str(mag_x)+';'+str(mag_y)+';'+str(mag_z)
# -------------------------------------------------------

# funzione per creare il nome del foto
def createNamePhoto():
    return('img_'+str(getNumerPhoto())+'.png')
# -------------------------------------------------------

# funzione per acquisizione immagine 
def makePhoto():
    nameUltimatePhoto = createNamePhoto()  # l'ho salvato dentro una variabile in modo che se l'immagine non andasse bene potesse essere eliminta facilmente
    camera.capture(nameUltimatePhoto)

    # Convert the color from RGB to Grayscale
    img = cv.imread(nameUltimatePhoto)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if(machineLearning(img_gray)==True):
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
def machineLearning(img_gray):
    return True
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