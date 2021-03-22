#!/bin/env python

import numpy as np


class geCsvAdapter:
    def __init__(self, dataFrame, positionConfig):
        self.df = dataFrame
        self.positionConfig = positionConfig

    def updateColumn(self):
        print "[Update] GE Csv in process..."
        self.updateSColumn()
        self.updateTColumn()

        # position Table configuration
        self.updateAIColumn()
        self.updateAJColumn()
        self.updateAKColumn()

        return self.df

    def updateTColumn(self):
        self.df["T"] = 0.30000000
        return

    def updateSColumn(self):
        self.df["S"] = "Copper or Copper compound"
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
