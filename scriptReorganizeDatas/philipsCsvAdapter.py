#!/bin/env python

import numpy as np
import pandas as pd


class philipsCsvAdapter:
    def __init__(self, dataFrame, positionConfig):
        self.df = dataFrame
        self.positionConfig = positionConfig

    def updateColumn(self):
        print "[Update] Philips Csv in process..."
        self.copyColumn("R")
        self.copyColumn("S")
        self.copyColumn("T")
        self.copyColumn("U")
        self.copyColumn("V")
        self.copyColumn("W")
        self.copyColumn("AG")
        self.updateADColumn()

        # position Table configuration
        self.updateAIColumn()
        self.updateAJColumn()
        self.updateAKColumn()
        self.calculAC()
        self.emptyAE()

        return self.df

    def is_nan(self, x):
        return (x is np.nan or x != x)

    def emptyAE(self):
        self.df["AE"] = 0

    def calculAC(self):
        idx = 0
        for cell in self.df["AC"]:
            aeCalc = float(self.df["AE"][idx])
            zCalc = float(self.df["Z"][idx])
            self.df["AC"][idx] = aeCalc * zCalc
            idx += 1
        return self.df

    def searchNextValue(self, column, index):
        if index == 0:
            return self.df[column].loc(self.df[column].first_valid_index())
        tmpFrame = self.df[column].copy()
        tmpIdx = 0
        if index > 0:
            while (tmpIdx < index):
                tmpFrame[tmpIdx] = None
                tmpIdx += 1
        return tmpFrame.loc(tmpFrame.first_valid_index())

    def copyColumn(self, column):
        tmpVal = self.df[column][0]
        if (self.is_nan(tmpVal) == True):
            counter = 0
            while (self.is_nan(tmpVal) == True):
                tmpVal = self.df[column][counter]
                counter += 1
        counter = 0
        for idx in self.df[column]:
            # If cell is empty and value is set
            if self.is_nan(idx) == True and self.is_nan(tmpVal) == False:
                self.df[column][counter] = tmpVal
            # else if tmpVal is null search next one
            elif tmpVal is None:
                tmpVal = self.searchNextValue(column, counter)
            elif idx is not None and tmpVal != idx:
                tmpVal = idx
            counter = counter + 1
        return self.df

    def updateADColumn(self):
        ctx = 0
        for cell in self.df["AD"]:
            if (self.df["G"][ctx] == "Fluoroscopy"):
                self.df["AD"][ctx] = 0.60
            else:
                self.df["AD"][ctx] = 1.000
            ctx += 1
        return

    # Value given in argument of the script
    def updateAGColumn(self):
        self.df["AG"] = self.df["AG"][0]
        return

    def updateAIColumn(self):
        self.df["AI"] = self.positionConfig["lo"]
        return

    def updateAJColumn(self):
        self.df["AJ"] = self.positionConfig["la"]
        return

    def updateAKColumn(self):
        self.df['AK'] = self.positionConfig["ht"]
        return
