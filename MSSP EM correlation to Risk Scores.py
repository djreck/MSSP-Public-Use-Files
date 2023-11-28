# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 08:05:16 2023

@author: djreck
"""
%reset -f
import pandas
import pyodbc
import psycopg2
import snowflake
import mysql
import matplotlib.pyplot as plt
import random
from sqlite3 import connect
import numpy
import os
from pathlib import Path
import glob
import requests
from pandas import json_normalize
from sklearn.linear_model import LinearRegression
import locale
from scipy import stats
import os.path
import sys
import tkinter
import tkinter.filedialog as fd
import os


#CMS Public Use Files prior to 2021 are currently unavailable
#Temporary solution: just bringing in the file from our drive. We'd downloaded this 2 years ago
#There's a popup box for other users to supply thier own filepath
PUF_2020 = "XX:/MSSP/Public Use Files/Trimmed PUFs/CSVs/2020_Shared_Savings_Program_PUF_TRIMMED.csv"

APIs = ['https://data.cms.gov/data-api/v1/dataset/73b2ce14-351d-40ac-90ba-ec9e1f5ba80c/data'
        ,'https://data.cms.gov/data-api/v1/dataset/bd6b766f-6fa3-43ae-8e9a-319da31dc374/data'
#       ,'https://data.cms.gov/data-api/v1/dataset/69ec2609-5ce5-4ce1-b14c-1f8809fda2c2/data'
        ]
data = ['Financial_Quality_22'
        ,'Financial_Quality_21'
#,'ACOs'
        ]

#Checking whether filepath exists to PUF_2020 as a csv and prompts user to enter path
fileCheck = os.path.exists(PUF_2020)
if fileCheck == False:
   
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    
    currdir = os.getcwd()
    tempdir = fd.askopenfilename(parent=root, initialdir=currdir, title='Please select your 2020 MSSP Financial and Quality Results Public Use File Saved as a csv')
    
    if len(tempdir) > 0:
        print ("You chose %s" % tempdir)
        if tempdir[-3:] != "csv":
            print ("Please save the file as a csv and try again")
            exit(1)
        else:
            PUF_2020 = tempdir
    else:
        exit(1)

d={}
for i in range(0,len(APIs)):
    response = requests.get(APIs[i])
    if response.status_code == 200:
        print("Successful connection with API.")
        print('-------------------------------')
        j=response.json()
        d[data[i]] = pandas.DataFrame.from_dict(j)
    elif response.status_code == 404:
        print("Unable to reach URL.")
    else:
        print("unable to connect API or retrieve data.")


df20 = pandas.read_csv(PUF_2020)
df20 = df20.dropna(axis=1,how='all')
d['Financial_Quality_20'] = pandas.DataFrame.from_dict(df20)

df3 = d['Financial_Quality_20']
df4 = d['Financial_Quality_21']
df5 = d['Financial_Quality_22']

df3['source'] = 2020
df4['source'] = 2021
df5['source'] = 2022

df3.set_index('ACO_ID')
df4.set_index('ACO_ID')
df5.set_index('ACO_ID')

#Getting all of the 2022 MSSP ACOs, will subset earlier years for these ACOs
ACOs = pandas.unique(df5['ACO_ID'])

subset20 = df3[df3['ACO_ID'].isin(ACOs)]
subset21 = df4[df4['ACO_ID'].isin(ACOs)]

#Adding years as suffixes for prior years to maintain clear identification for columns
subset20 = subset20.add_suffix('_2020')
subset21 = subset21.add_suffix('_2021')
result = pandas.concat([df5,subset21,subset20], axis = 1, join = "inner")

#cleaning EM and Risk Score fields for calculations
result['P_EM_Total'] = result['P_EM_Total'].astype(float)
result['P_EM_Total_2021'] = result['P_EM_Total_2021'].str.replace(",","").astype(float)
result['P_EM_Total_2020'] = result['P_EM_Total_2020'].astype(float)
result['CMS_HCC_RiskScore_AGND_PY'] = result['CMS_HCC_RiskScore_AGND_PY'].astype(float)
result['CMS_HCC_RiskScore_AGND_PY_2021'] = result['CMS_HCC_RiskScore_AGND_PY_2021'].astype(float)

#specialists only
result['P_EM_SP_Vis'] = result['P_EM_SP_Vis'].astype(float)
result['P_EM_SP_Vis_2021'] = result['P_EM_SP_Vis_2021'].str.replace(",","").astype(float)
result['P_EM_SP_Vis_2020'] = result['P_EM_SP_Vis_2020'].astype(float)

#PCPs only
result['P_EM_PCP_Vis'] = result['P_EM_PCP_Vis'].astype(float)
result['P_EM_PCP_Vis_2021'] = result['P_EM_PCP_Vis_2021'].str.replace(",","").astype(float)
result['P_EM_PCP_Vis_2020'] = result['P_EM_PCP_Vis_2020'].astype(float)

#adding columns to calculate the differences
result['EM_20_to_21'] = result['P_EM_Total_2021'] - result['P_EM_Total_2020']
result['EM_SP_20_to_21'] = result['P_EM_SP_Vis_2021'] - result['P_EM_SP_Vis_2020']
result['EM_PCP_20_to_21'] = result['P_EM_PCP_Vis_2021'] - result['P_EM_PCP_Vis_2020']
result['AGND_HCC_21_to_22'] = result['CMS_HCC_RiskScore_AGND_PY'] - result['CMS_HCC_RiskScore_AGND_PY_2021']

#Setting the change in Aged Non-Dual risk score to be the outcome variable
y = result['AGND_HCC_21_to_22'].values.reshape(-1,1)

#Setting total, PCP, and specialist primary care visits as the potential explanatory variables
xVars = ['EM_20_to_21',
        'EM_SP_20_to_21',
        'EM_PCP_20_to_21']
desc = ['All EM',
        'Specialist EM',
        'PCP EM']

#This loop creates the linear regression models and plots for each explanatory variable in xVars    
for i in range(0,len(xVars)):
    x = result[xVars[i]].values.reshape(-1,1)
    #creating lm and getting stats
    linear_regressor=LinearRegression().fit(x,y)
    y_pred = linear_regressor.predict(x)
    r = linear_regressor.score(x,y)
    intercept = linear_regressor.intercept_.item()
    slope = linear_regressor.coef_.item()
    #plot of lm with formula
    plt.title("2022 MSSP ACOs\nAged Non-Dual HCC Risk Score Delta ('21 to '22) vs \nE&M Utilization Delta ('20 to '21)")
    plt.scatter(x,y)
    plt.plot(x,y_pred, color = 'red')
    #plt.axhline(y=numpy.mean(y),linestyle = '--')
    #plt.annotate("Average: "+str(round(numpy.mean(y),4)),(8000,numpy.mean(y)))
    plt.axvline(x=numpy.mean(x),linestyle = '--')
    plt.annotate("Average: "+str(round(numpy.mean(x),2)),(numpy.mean(x),-0.8))
    plt.annotate("r-squared = "+str(round(r,4)),(3000,0.8))
    plt.annotate("y = "+str(round(slope,6))+"x +"+str(round(intercept,2)),(3000,1.00))
    plt.xlabel("Change in "+r"$\bf{" + desc[i] + "}$" +" Visits per 1,000 \n2020 to 2021")
    plt.ylabel("Change in AGND HCC Risk Score \n2021 to 2022")
    plt.show()

df_explore = result[['ACO_ID',
                     'ACO_Name',
                     'P_EM_Total_2021',
                     'P_EM_Total_2020',
                     'EM_20_to_21',
                     'P_EM_SP_Vis_2021',
                     'P_EM_SP_Vis_2020',
                     'EM_SP_20_to_21',
                     'P_EM_PCP_Vis_2021',
                     'P_EM_PCP_Vis_2020',
                     'EM_PCP_20_to_21',
                     'CMS_HCC_RiskScore_AGND_PY',
                     'CMS_HCC_RiskScore_AGND_PY_2021',
                     'AGND_HCC_21_to_22']]