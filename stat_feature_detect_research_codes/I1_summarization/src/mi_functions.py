import sys
import re
import os
import numpy as np
import cmath
import random
import vtk

#################################################################################
## Compute specific mutual information and pointwise mutual information measures
#################################################################################
def compute_specific_mutual_information(Array1,Array2,ArrayComb,numSamples,bins):

    I11 = np.zeros(bins)
    I12 = np.zeros(bins)
    I21 = np.zeros(bins)
    I22 = np.zeros(bins)
    I31 = np.zeros(bins)
    I32 = np.zeros(bins)

    prob_of_x_given_y=0.0
    prob_of_y_given_x=0.0
    prob_of_x=0.0
    prob_of_y=0.0

    for i in range(0,bins):
        for j in range(0,bins):
            if Array1[i] == 0:
                prob_of_y_given_x=0
            else:
                prob_of_y_given_x = float(ArrayComb[i][j]) / float(Array1[i])

            prob_of_y = float(Array2[j]) / numSamples

            if prob_of_y_given_x != 0 and prob_of_y != 0:
                I11[i] =  I11[i] + prob_of_y_given_x * np.log2(prob_of_y_given_x / prob_of_y)
                
            if prob_of_y_given_x != 0:
                I21[i] = I21[i] + prob_of_y_given_x * np.log2(prob_of_y_given_x)
                
            if prob_of_y != 0:
                I21[i] =  I21[i] - prob_of_y * np.log2(prob_of_y)
                
            if(Array2[i] == 0):
                prob_of_x_given_y = 0
                
            else:
                prob_of_x_given_y = float(ArrayComb[j][i]) / Array2[i]; 

            prob_of_x = float(Array1[j]) / numSamples

            if prob_of_x_given_y != 0 and prob_of_x != 0:
                I12[i] = I12[i] + prob_of_x_given_y * np.log2(prob_of_x_given_y / prob_of_x)

            if(prob_of_x_given_y != 0):
                I22[i] = I22[i] + prob_of_x_given_y * np.log2(prob_of_x_given_y)

            if(prob_of_x != 0):
                I22[i] = I22[i] - prob_of_x * np.log2(prob_of_x)

            if(prob_of_y_given_x > 1.0):
                print ("Value of prob_of_y_given_x is greater than 1")

            if(prob_of_x_given_y > 1.0):
                print ("Value of prob_of_x_given_y is greater than 1")

    for i in range(0,bins):
        for j in range(0,bins):
            if Array1[i] == 0:
                prob_of_y_given_x=0
            else:
                prob_of_y_given_x = float(ArrayComb[i][j]) / Array1[i]

            prob_of_y = float(Array2[j]) / numSamples

            I31[i] = I31[i] + prob_of_y_given_x * I22[j]

            if(Array2[i] == 0):
                prob_of_x_given_y = 0
            else:
                prob_of_x_given_y = float(ArrayComb[j][i]) / Array2[i] 

            prob_of_x = float(Array1[j]) / numSamples
            I32[i] = I32[i] + prob_of_x_given_y * I21[j]
            
    return I11,I12,I21,I22,I31,I32

#################################################################################
## Compute poitwise mutual information
#################################################################################
def compute_pointwise_mutual_information(Array1,Array2,ArrayComb,numSamples,bins):
    pmi_array = np.zeros_like(ArrayComb)
    
    for i in range(bins):
        for j in range(bins):
            
            if ArrayComb[i][j]==0:
                pmi_array[i][j]=0
            elif ArrayComb[i][j] > 0 and Array1[i] > 0  and Array2[j] > 0:
                prob_x = Array1[i]/float(numSamples)
                prob_y = Array2[j]/float(numSamples)
                joint_prob_xy = ArrayComb[i][j]/float(numSamples)
                pmi_array[i][j] = np.log2(joint_prob_xy/(prob_x*prob_y))
            else:
                pmi_array[i][j]=0
             
            ## normalize betweem [-1,1]
            #if pmi_array[i][j] != 0:
            #    pmi_array[i][j] = pmi_array[i][j]/(-np.log2(joint_prob_xy))
                
    return pmi_array   

#################################################################################
## Compute self information 1/log(p(x))
#################################################################################
def compute_self_information(Array1,numSamples,bins):
    SI = np.zeros(bins)
    prob_of_x=0.0

    for i in range(0,bins):
        prob_of_x = float(Array1[i]) / numSamples

        if(prob_of_x>0):
            SI[i] = -np.log2(prob_of_x)

    return SI   


#################################################################################
## Compute I1 (surprise SMI)
#################################################################################
def compute_I1(Array1,Array2,ArrayComb,numSamples,bins):

    I11 = np.zeros(bins)
    I12 = np.zeros(bins)

    prob_of_x_given_y=0.0
    prob_of_y_given_x=0.0
    prob_of_x=0.0
    prob_of_y=0.0

    for i in range(0,bins):
        for j in range(0,bins):
            if Array1[i] == 0:
                prob_of_y_given_x=0
            else:
                prob_of_y_given_x = float(ArrayComb[i][j]) / float(Array1[i])

            prob_of_y = float(Array2[j]) / numSamples

            if prob_of_y_given_x != 0 and prob_of_y != 0:
                I11[i] =  I11[i] + prob_of_y_given_x * np.log2(prob_of_y_given_x / prob_of_y)
                
                
            if(Array2[i] == 0):
                prob_of_x_given_y = 0
                
            else:
                prob_of_x_given_y = float(ArrayComb[j][i]) / Array2[i]; 

            prob_of_x = float(Array1[j]) / numSamples

            if prob_of_x_given_y != 0 and prob_of_x != 0:
                I12[i] = I12[i] + prob_of_x_given_y * np.log2(prob_of_x_given_y / prob_of_x)

            if(prob_of_y_given_x > 1.0):
                print ("Value of prob_of_y_given_x is greater than 1")

            if(prob_of_x_given_y > 1.0):
                print ("Value of prob_of_x_given_y is greater than 1")

           
    return I11,I12
             