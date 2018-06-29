# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:35:35 2017

@author: krecinic
"""

import os 
import os.path
import numpy as np

import scipy.special 
import scipy.optimize 
import FitFunctions as ff

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import time

import sys,pdb

class DaXView(pg.ImageView):
    def __init__(self, parent=None):
        pg.ImageView.__init__(self,parent,view=pg.PlotItem())
        
        # Override some original ImageView UI elements
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.splitter.setSizes([self.height()-60,60])
        
        # Axes display settings        
        self.view.invertY(b=False)
        self.view.setLabel('bottom',text='X',units='m')
        self.view.setLabel('left',text='Y',units='m')
        self.view.setLabel('top',units='m')
        self.view.setLabel('right',units='m')
        self.view.showAxis('top')
        self.view.showAxis('right')
        
        self.ui.roiPlot.setLabel('bottom',text='Time')
        self.ui.roiPlot.setLabel('left',text='Intensity (arb.u.)')
        self.ui.roiPlot.setMinimumHeight(60)
        self.ui.roiPlot.setMaximumHeight(60)
        
        # IntX, IntY plot
        self.roi2DPlot = pg.PlotWidget()
        self.roi2DCurve = self.roi2DPlot.plot()
        self.ui.splitter.addWidget(self.roi2DPlot)
        self.roiIntX=False
        self.roiIntY=False
        self.sigTimeChanged.connect(self.updateRoi2DPlot)
        self.roi2DPlot.setLabel('bottom',text='Length',units='m')
        self.roi2DPlot.setLabel('left',text='Intensity (arb.u.)')
        self.roi2DPlot.hide()
        
        # Add few extra handles to the ROI to make it more line-like
        self.roi.addScaleRotateHandle([0, 0.5], [1, 0.5])
        self.roi.addScaleRotateHandle([1, 0.5], [0, 0.5])
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])

        self.dirname='/'
        
        self.scale=1.0

        self.normint=False
        self.diffimg=False     
        self.diffRgn = pg.LinearRegionItem()
        self.diffRgn.setZValue(0)
        self.diffRgn.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))
        self.ui.roiPlot.addItem(self.diffRgn)
        self.diffRgn.hide()
        self.sigprox=pg.SignalProxy(self.diffRgn.sigRegionChanged, slot=self.normChanged)
        
        self.fitRgn = pg.LinearRegionItem()
        self.fitRgn.setZValue(0)
        self.ui.roiPlot.addItem(self.fitRgn)
        self.fitRgn.hide()

        ## Create random 3D data set with noisy signals
        img = pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 20 + 100
        img = img[np.newaxis,:,:]
        decay = np.exp(-np.linspace(0,0.3,100))[:,np.newaxis,np.newaxis]
        self.data = np.random.normal(size=(100, 200, 200))
        self.data += img * decay
        self.data += 2
        
        ## Add time-varying signal
        sig = np.zeros(self.data.shape[0])
        sig[30:] += np.exp(-np.linspace(1,10, 70))
        sig[40:] += np.exp(-np.linspace(1,10, 60))
        sig[70:] += np.exp(-np.linspace(1,10, 30))
        
        sig = sig[:,np.newaxis,np.newaxis] * 3
        self.data[:,50:60,50:60] += sig
        
        ## Display the data and assign each frame a time value from 1.0 to 3.0
        self.Tbegin=0.
        self.DeltaT=10.
        self.tvals=np.linspace(0., 10., self.data.shape[0])
        self.setImage(self.data, xvals=self.tvals)    
        
        self.roidata=None
        
        self.roi1Ddata=None
        self.roiIntXdata=None
        self.roiIntYdata=None
        
        
        # Create fit curve
        self.fitCurve = self.ui.roiPlot.plot()
        self.fitCurve2D = self.roi2DPlot.plot()
        self.fitdata = []
        self.roiIntXYcoords= np.zeros(10)
        
        # Set curve symbols and colors        
        self.roiCurve.setPen(pg.mkPen(color='r',width=2))        
        self.roiCurve.setSymbol('o')
        self.roiCurve.setSymbolPen(pg.mkPen(color='r'))
        self.roiCurve.setSymbolBrush(pg.mkBrush(color=pg.mkColor(255,0,0,60)))
        self.fitCurve.setPen(pg.mkPen(color='b',width=2))      
        
        
        self.roi2DCurve.setPen(pg.mkPen(color='g',width=2))        
        self.roi2DCurve.setSymbol('d')
        self.roi2DCurve.setSymbolPen(pg.mkPen(color='g'))
        self.roi2DCurve.setSymbolBrush(pg.mkBrush(color=pg.mkColor(0,0,255,60)))
        self.fitCurve2D.setPen(pg.mkPen(color='b',width=2))   
        
        self.looplay=False
        self.playspeed=10

      
    def readBin(self,path,width,height):
        return np.reshape(np.fromfile(path,'>d',count=width*height),
                          (width,height))
        
    def readSequence(self,dirname,width,height):
        # List all files in target directory
        QtCore.QDir.setCurrent(dirname)
        fnames=sorted(os.listdir(dirname))
        # Filter out .bin files only
        binf=[]
        for f in fnames:
            if f.endswith('.bin'):
                binf.append(f)
        self.dirname=dirname
        # If there were no readable files found return immediately
        if binf==[]:
            return
        # Read in all binary files
        progress=QtGui.QProgressDialog('Reading sequence...','Abort',
                                       0,len(binf),self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        frames=[]
        for i in range(len(binf)):
            frames.append(self.readBin(binf[i],width,height))
            progress.setValue(i)
            QtGui.QApplication.processEvents()
            if progress.wasCanceled():
                return
        progress.close()
        self.data=np.array(frames)
        self.rawdata=np.copy(self.data)
        # Check time axis 
        npts=self.data.shape[0]
        if npts != len(self.tvals):
            # Update time axis and image data
            self.setTaxis(self.Tbegin,self.DeltaT)
            # Update ROI 
            self.roiChanged()
        else:
            # Update image data only
            self.setImage(self.data, xvals=self.tvals)

    def setSpaceAxis(self,pixsize,mag,hide=False):
        self.scale=pixsize/mag
        self.view.getAxis('left').setScale(scale=self.scale)
        self.view.getAxis('right').setScale(scale=self.scale)
        self.view.getAxis('bottom').setScale(scale=self.scale)
        self.view.getAxis('top').setScale(scale=self.scale)
        if hide:
            self.view.hideAxis('left')
            self.view.hideAxis('bottom')
            self.view.hideAxis('right')
            self.view.hideAxis('top')
        else:
            self.view.showAxis('left')
            self.view.showAxis('bottom')
            self.view.showAxis('right')
            self.view.showAxis('top')

        (ind, time) = self.timeIndex(self.timeLine)
        self.updateRoi2DPlot(ind,time)

    def setTaxis(self,tbegin,deltat,unit='None'):
        self.Tbegin=tbegin
        self.DeltaT=deltat
        npts=self.data.shape[0]
        self.tvals=np.linspace(tbegin,tbegin+deltat*(npts-1),npts)
        if unit == 'None':
            self.ui.roiPlot.setLabel('bottom',text='Time')
        else:
            self.ui.roiPlot.setLabel('bottom',text='Time'+' ('+unit+')')
        self.setImage(self.data, xvals=self.tvals)
    
    def setFitFunction(self,fittype,fitpars):      
        # Update the curve and show/hide depending on the type parameter
        if fittype == 'None':
            self.fitCurve.hide()
        else:
            if self.roiIntX or self.roiIntY:
                self.fitdata=ff.funcs[fittype](self.roiIntXYcoords,fitpars)
                self.fitCurve2D.setData(y=self.fitdata, x=self.roiIntXYcoords)
                self.fitCurve2D.show()
            else:
                self.fitdata=ff.funcs[fittype](self.tvals,fitpars)
                self.fitCurve.setData(y=self.fitdata, x=self.tvals)
                self.fitCurve.show()
        
    def fit(self,fittype,fitpars):
        if fittype != 'None':
            (ind, time) = self.timeIndex(self.timeLine)
            if self.roiIntX:
                if self.roiIntXdata is not None:
                    self.roiIntXYcoords = np.array(range(len(self.roiIntXdata[0])))*self.scale
                    xdata=self.roiIntXYcoords
                    ydata=self.roiIntXdata[ind]
            elif self.roiIntY:
                if self.roiIntYdata is not None:
                    self.roiIntXYcoords = np.array(range(len(self.roiIntYdata[0])))*self.scale
                    xdata=self.roiIntXYcoords
                    ydata=self.roiIntYdata[ind]
            else:
                if self.fitRgn.isVisible():
                    sind,st=self.timeIndex(self.fitRgn.lines[0])
                    eind,et=self.timeIndex(self.fitRgn.lines[1])
                    xdata=self.tvals[sind:eind+1]
                    ydata=self.roi1Ddata[sind:eind+1]
                    if len(xdata) <= len(fitpars):
                        return fitpars,'Fit range too small: not enough data points.'
                else:
                    xdata=self.tvals
                    ydata=self.roi1Ddata
            fitpars=fitpars[0:ff.funcspnum[fittype]]
            
            # Perform fit
            info=''
            try:
                popt,cov,infodict,mesg,success=\
                scipy.optimize.leastsq(ff.lsqfuncs[fittype],fitpars,
                                                    args=(xdata,ydata),full_output=1)
            except TypeError as err:
                return fitpars,str(err)
            except RuntimeWarning as warn:
                info+=str(warn)+'\n'
            
            # Return some useful info/parameters
            if success > 4 or success < 1:
                info+='Fitting may not have converged.\n'
            if fittype == 'Erf':
                info += 'FWHM='+str(2*np.sqrt(np.log(2))*popt[3])+'\n'
            if fittype == 'Gaussian':
                info += 'FWHM='+str(2*np.sqrt(2*np.log(2))*popt[3])+'\n'
        return popt,info
        
    def normChanged(self):
        # Store settingss
        if self.diffimg:
            self.diffRgn.show()
        else:
            self.diffRgn.hide()
        # Trigger image update etc.
        self.imageDisp = None
        self.updateImage()
        self.autoLevels()
        self.roiChanged()
        self.sigProcessingChanged.emit(self)
            
    def normalize(self, image):
        if self.normint==False and self.diffimg==False:
            return image
            
        norm = image.view(np.ndarray).copy()
            
        if self.normint and image.ndim == 3:
            n = image.mean(axis=1).mean(axis=1)
            n.shape = n.shape + (1, 1)
            norm /= n
        
        if self.diffimg and image.ndim == 3:
            (sind, start) = self.timeIndex(self.diffRgn.lines[0])
            (eind, end) = self.timeIndex(self.diffRgn.lines[1])
            #print start, end, sind, eind
            n = image[sind:eind+1].mean(axis=0)
            n.shape = (1,) + n.shape
            norm -= n
            
        return norm
    
    def roiTypeSelected(self,roitype):
        if roitype == 'Total':
            self.roiIntX = False
            self.roiIntY = False
            self.roi2DPlot.hide()
            self.ui.roiBtn.setChecked(True)
            self.ui.roiPlot.setMaximumHeight(60000)
            self.roiClicked()
        elif roitype == 'IntX':
#            QtCore.pyqtRemoveInputHook()
#            pdb.set_trace()
            self.roiIntX = True
            self.roiIntY = False
            self.ui.roiBtn.setChecked(False)
            self.roiClicked()
            self.ui.roiPlot.setMaximumHeight(60)
            self.roi2DPlot.show()
            self.roi.show()
            self.ui.splitter.setSizes([(self.height()-60)*0.6,60,
                                       (self.height()-60)*0.4])
            (ind, time) = self.timeIndex(self.timeLine)
            self.updateRoi2DPlot(ind,time)
        elif roitype == 'IntY':
            self.roiIntX = False
            self.roiIntY = True
            self.ui.roiBtn.setChecked(False)
            self.roiClicked()
            self.ui.roiPlot.setMaximumHeight(60)
            self.roi2DPlot.show()
            self.roi.show()
            self.ui.splitter.setSizes([(self.height()-60*0.6),60,
                                       (self.height()-60*0.4)])
            (ind, time) = self.timeIndex(self.timeLine)
            self.updateRoi2DPlot(ind,time)
        elif roitype == 'Tot. & IntX':
            self.roiIntX = True
            self.roiIntY = False
            self.ui.roiBtn.setChecked(True)
            self.ui.roiPlot.setMaximumHeight(60000)
            self.roiClicked()
            self.roi2DPlot.show()
            self.roi.show()
            self.ui.splitter.setSizes([self.height()*0.4,self.height()*0.3,
                                       self.height()*0.3])
            (ind, time) = self.timeIndex(self.timeLine)
            self.updateRoi2DPlot(ind,time)            
        else:
            self.roiIntX = False
            self.roiIntY = False
            self.roi2DPlot.hide()
            self.ui.roiBtn.setChecked(False)
            self.roiClicked()
            self.ui.roiPlot.setMaximumHeight(60)
            self.ui.splitter.setSizes([self.height()-60,60])
            
    def roiChanged(self):
        if self.image is None:
            return
            
        image = self.getProcessedImage()
           
        if image.ndim == 2:
            axes = (0, 1)
        elif image.ndim == 3:
            axes = (1, 2)
        else:
            return
        
        # Get ROI data
        self.roidata, self.roicoords = self.roi.getArrayRegion(image.view(np.ndarray), self.imageItem, axes, returnMappedCoords=True)
        
        # Integrate and display 
        if self.roidata is not None:
            self.roi1Ddata=self.roidata
            while self.roi1Ddata.ndim > 1:
                self.roi1Ddata = self.roi1Ddata.mean(axis=1)
            if image.ndim == 3:
                self.roiCurve.setData(y=self.roi1Ddata, x=self.tVals)
                # Integrate in X or Y direction
                self.roiIntXdata=self.roidata.mean(axis=1)
                self.roiIntYdata=self.roidata.mean(axis=2)
                # Set the data to roi2DCurve
                (ind, time) = self.timeIndex(self.timeLine)
                self.updateRoi2DPlot(ind,time)
            else:
                while self.roicoords.ndim > 2:
                    self.roicoords = self.roicoords[:,:,0]
                self.roicoords = self.roicoords - self.roicoords[:,0,np.newaxis]
                xvals = (self.roicoords**2).sum(axis=0) ** 0.5
                self.roiCurve.setData(y=self.roi1Ddata, x=xvals)
            
    def updateRoi2DPlot(self,ind,time):
        if self.roiIntX:
            if self.roiIntXdata is not None:
                self.roiIntXYcoords = np.array(range(len(self.roiIntXdata[0])))*self.scale
                self.roi2DCurve.setData(y=self.roiIntXdata[ind],x=self.roiIntXYcoords)
        else:
            if self.roiIntYdata is not None:
                self.roiIntXYcoords = np.array(range(len(self.roiIntYdata[0])))*self.scale
                self.roi2DCurve.setData(y=self.roiIntYdata[ind],x=self.roiIntXYcoords)
    
    def setCurrentIndex(self,ind):
        pg.ImageView.setCurrentIndex(self,ind)
        ind, time = self.timeIndex(self.timeLine)
        self.updateRoi2DPlot(ind,time)

    def keyPressEvent(self, ev):
#        print ev.key()
        if ev.key() == QtCore.Qt.Key_Space:
            if self.playTimer.isActive() == False:
                self.play(self.playspeed)
            else:
                self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_Home:
            self.setCurrentIndex(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.setCurrentIndex(self.getProcessedImage().shape[0]-1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
        else:
            QtGui.QWidget.keyPressEvent(self, ev)
    
#    def play(self,rate):
#        self.playRate = rate
#        if rate == 0:
#            self.playTimer.stop()
#            return
#            
#        self.lastPlayTime = time.time()
#        if not self.playTimer.isActive():
#            self.playTimer.start(16)
    
    def play(self, rate):
        self.playRate = rate
        if rate == 0:
            self.playTimer.stop()
            return
            
        self.lastPlayTime = time.time()
        if not self.playTimer.isActive():
            self.playTimer.start(16)
    
    def timeout(self):
        now = time.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return
        n = int(self.playRate * dt)
        if n != 0:
            self.lastPlayTime += (float(n)/self.playRate)
            if self.currentIndex+n >= self.image.shape[0]:
                self.play(0)
                if self.looplay:
                    QtCore.QTimer.singleShot(16,self.replay)
            self.jumpFrames(n)
            
    def replay(self):
        self.setCurrentIndex(0)
        self.play(self.playspeed)
            
#                QtCore.pyqtRemoveInputHook()
#                pdb.set_trace()
            