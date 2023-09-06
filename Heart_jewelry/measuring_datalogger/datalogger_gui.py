import PySimpleGUI as sg
import numpy as np
from datetime import *


def get_date_time():

    #now = datetime.now(timezone.utc)   # datetime object containing current date and time
    now = datetime.now(timezone(timedelta(hours=2),'utc'))
    
    day = int(now.strftime("%d"))
    month = int(now.strftime("%m"))
    year = int(now.strftime("%Y"))
    hour = int(now.strftime("%H"))
    minu = int(now.strftime("%M"))
    sec = int(now.strftime("%S"))
    ms = int(now.strftime("%f"))  # do not need for heart data
    
    #print('day : ' + str(day))
    #print('month : ' + str(month))
    #print('year : ' + str(year))
    #print('hour : ' + str(hour))
    #print('minu : ' + str(minu))
    #print('sec : ' + str(sec))
    #print('ms : ' + str(ms))
    
    return day, month, year, hour, minu, sec
    
    
def START_STOP(bool_input, hour, minu, sec):
    if bool_input == True:  # Checked : Reporting a start time
        STARThour = hour
        STARTminu = minu
        STARTsec = sec
        
        # Way 2 : manual input
        #STARThour = np.array(values['-IN-'])  # can only put numbers
        #STARTminu = np.array(values['-IN2-'])  # can only put numbers
        #STARTsec = 0
        
        #Put STOP time to zero: because reporting START time only
        STOPhour = 0 
        STOPminu = 0
        STOPsec = 0
    else:
        STARThour = 0 
        STARTminu = 0
        STARTsec = 0
        STOPhour = hour 
        STOPminu = minu
        STOPsec = sec

    return STARThour, STARTminu, STARTsec, STOPhour, STOPminu, STOPsec


def algo(datamat):
    day, month, year, hour, minu, sec = get_date_time()
    
    bool_input = values['-IN3-']
    STARThour, STARTminu, STARTsec, STOPhour, STOPminu, STOPsec = START_STOP(bool_input, hour, minu, sec)
    
    newvals = [[day, month, year, STARThour, STARTminu, STARTsec, STOPhour, STOPminu, STOPsec, stim]]
    out = np.concatenate((datamat, newvals), axis=0)
    np.savetxt("out.txt", out, fmt="%s")
    datamat = np.loadtxt("out.txt")
    
    return datamat


# Load the output matrix data, for appending the input 
datamat = np.loadtxt("out.txt")

#sg.theme("DarkBlue15")
sg.theme("LightGreen1")
#sg.theme("reddit")
#sg.theme("DarkTeal2")
#sg.theme("Black")
#sg.theme("TealMono")
#sg.theme("TanBlue")
#sg.theme("Tan")

# Adding a radio button
#[sg.Radio('Permission Granted', "RADIO1", default=False)]

#layout = [[sg.Text("Log your NEXT stimuli!")], [sg.Button("EXIT")], [sg.Input(key='-IN-')], [sg.Button("Submit")]]

#layout = [[sg.Text("Log your NEXT stimuli!")], [sg.Button("Rose")],  [sg.Button("Howlite")], [sg.Button("Metal")], [sg.Button("None")], [sg.Input(key='-IN-')], [sg.Input(key='-IN2-')], [sg.Button("EXIT")]]

layout = [[sg.Text("Log your NEXT stimuli! CHECKED=report a START time, UNCHECKED=report a STOP time.")], [sg.Checkbox('CHECKED=report a START time:', default=True, key="-IN3-")], [sg.Button("Rose")],  [sg.Button("Howlite")], [sg.Button("Metal")], [sg.Button("No bracelet")], [sg.Button("Bain")], [sg.Button("PC")]]

# Create the window
#window = sg.Window("DataLogger", layout)
# OR
window = sg.Window("DataLogger", layout, margins=(200, 200))
# OR
# sg.Window(title="DataLogger", layout=[[]], margins=(300, 300)).read()

# read() returns any events that are triggered in the Window() as a string as well as a values dictionary.

# Create an event loop
while True:
    event, values = window.read()
    
    if event == "Rose":  
        stim = 1
        datamat = algo(datamat)
        
    if event == "Howlite":
        stim = 2
        datamat = algo(datamat)
        
    if event == "Metal":
        stim = 3
        datamat = algo(datamat)
        
    if event == "No bracelet":
        stim = 4
        datamat = algo(datamat)
        
    if event == "Bain":
        stim = 5
        datamat = algo(datamat)
        
    if event == "PC":
        stim = 6
        datamat = algo(datamat)
        
    if event == sg.WIN_CLOSED:
        break

window.close()