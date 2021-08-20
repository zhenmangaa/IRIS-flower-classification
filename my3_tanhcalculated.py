# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:46:26 2020

@author: user
"""
import numpy as np
import math as mt


#def printbi(a, i ,j):
#    global k
#    if a % 2 ==1:
#        print("VW1%d%d%d W1%d%d%d 0 PWL(1n 0, 1.1n mvpp, 18n mvpp, 18.1n 0, 40.2n 0, 40.3n mvpp, 159.9n mvpp, 160n 0 , 171n 0, 171.1n mvpp)" % (i, j, k, i, j, k))
#    else:
#        print("VW1%d%d%d W1%d%d%d 0 PWL(1n 0, 1.1n mvpp, 18n mvpp, 18.1n 0, 171n 0, 171.1n mvpp)" % (i, j, k, i, j, k))
#    if k == 4:
#        return;
#    else:
#        k=+1
#    printbi(a % 2, i, j)

if __name__ == '__main__':
    ni = 4 #int(input("Please input ni :"))
    nh = 5 #int(input("Please input nh :"))
    no = 3 #int(input("Please input no :"))
    
    wi = np.loadtxt("w1.txt")
    wo = np.loadtxt("w2.txt")
    
    for i in range(len(wi)):
        print((wi[i]))
            
    print()
            
    for i in range(len(wo)):
        print(wo[i])
        
    print("//************layer1*************\n")
    for i in range(nh):
         print("//**********neuron1%d***********\n" % (i + 1))
         for j in range(ni + 1):
            if wi[j][i]>=0:
                print("VSIGN1<%d> SIGN1<%d> 0 DC mvdd\n" % (5*j + i + 1, 5*j + i + 1))
            else:
                print("VSIGN1<%d> SIGN1<%d> 0 DC 0\n" % (5*j + i + 1, 5*j + i + 1))

            if abs(wi[j][i]) == 1:
                print(
                    "VWP1<%d> WP1<%d> 0 PWL(0 0, 9.9n 0, 10n mvdd, 30n mvdd, 30.001n 0, 31.001n 0, 31.002n mvdd , 36n mvdd, 36.1n 0)" % (
                        5 * j + i + 1, 5 * j + i + 1))
                print(
                    "VWN1<%d> WN1<%d> 0 PWL(0 mvdd, 9.9n mvdd, 10n 0, 30n 0, 30.001n mvdd, 31.001n mvdd, 31.002n 0 , 36n 0, 36.1n mvdd)" % (
                        5 * j + i + 1, 5 * j + i + 1))
            elif wi[j][i] != 0:
                print(
                    "VWP1<%d> WP1<%d> 0 PWL(0 0, 9.9n 0, 10n mvdd, 30n mvdd, 30.001n 0, %.3fn 0, %.3fn mvdd , 36n mvdd, 36.1n 0)" % (
                    5 * j + i + 1, 5 * j + i + 1, 30.001 + 0.168 * mt.atanh(abs(wi[j][i]) ),
                    30.002 + 0.168 * mt.atanh(abs(wi[j][i]) )))
                print(
                    "VWN1<%d> WN1<%d> 0 PWL(0 mvdd, 9.9n mvdd, 10n 0, 30n 0, 30.001n mvdd, %.3fn mvdd, %.3fn 0 , 36n 0, 36.1n mvdd)" % (
                    5 * j + i + 1, 5 * j + i + 1, 30.001 + 0.168 * mt.atanh(abs(wi[j][i]) ),
                    30.002 + 0.168 * mt.atanh(abs(wi[j][i]) )))
            elif wi[j][i] == 0:
                print(
                    "VWP1<%d> WP1<%d> 0 PWL(0 0, 9.9n 0, 10n mvdd, 36n mvdd, 36.1n 0)" % (
                        5 * j + i + 1, 5 * j + i + 1 ))
                print(
                    "VWN1<%d> WN1<%d> 0 PWL(0 mvdd, 9.9n mvdd, 10n 0, 36n 0, 36.1n mvdd)" % (
                        5 * j + i + 1, 5 * j + i + 1))
            print()


    print("//************layer2*************\n")
    for i in range(no):
         print("//**********neuron2%d***********\n" % (i + 1))
         for j in range(nh + 1):
            if wo[j][i]>=0:
                print("VSIGN2<%d> SIGN2<%d> 0 DC mvdd\n" % (3*j + i + 1, 3*j + i + 1))
            else:
                print("VSIGN2<%d> SIGN2<%d> 0 DC 0\n" % (3*j + i + 1, 3*j + i + 1))

            if abs(wo[j][i]) == 1:
                print(
                    "VWP2<%d> WP2<%d> 0 PWL(0 0, 9.9n 0, 10n mvdd, 105n mvdd, 105.001n 0, 106.001n 0, 106.002n mvdd , 112n mvdd, 112.1n 0)" % (
                        3 * j + i + 1, 3 * j + i + 1))
                print(
                    "VWN2<%d> WN2<%d> 0 PWL(0 mvdd, 9.9n mvdd, 10n 0, 105n 0, 105.001n mvdd, 106.001n mvdd, 106.002n 0 , 112n 0, 112.1n mvdd)" % (
                        3 * j + i + 1, 3 * j + i + 1))
            elif wo[j][i]!=0:
                print(
                    "VWP2<%d> WP2<%d> 0 PWL(0 0, 9.9n 0, 10n mvdd, 105n mvdd, 105.001n 0, %.3fn 0, %.3fn mvdd , 112n mvdd, 112.1n 0)" % (
                        3 * j + i + 1, 3 * j + i + 1, 105.001 + 0.168 * mt.atanh(abs(wo[j][i]) ),
                        105.002 + 0.168 * mt.atanh(abs(wo[j][i]) )))
                print(
                    "VWN2<%d> WN2<%d> 0 PWL(0 mvdd, 9.9n mvdd, 10n 0, 105n 0, 105.001n mvdd, %.3fn mvdd, %.3fn 0 , 112n 0, 112.1n mvdd)" % (
                        3 * j + i + 1, 3 * j + i + 1, 105.001 + 0.168 * mt.atanh(abs(wo[j][i]) ),
                        105.002 + 0.168 * mt.atanh(abs(wo[j][i]) )))
            elif wo[j][i]==0:
                print(
                    "VWP2<%d> WP2<%d> 0 PWL(0 0, 9.9n 0, 10n mvdd, 112n mvdd, 112.1n 0)" % (
                        3 * j + i + 1, 3 * j + i + 1))
                print(
                    "VWN2<%d> WN2<%d> 0 PWL(0 mvdd, 9.9n mvdd, 10n 0, 112n 0, 112.1n mvdd)" % (
                        3 * j + i + 1, 3 * j + i + 1))
            print()
