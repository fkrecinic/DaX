# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 09:56:24 2017

@author: krecinic
"""

from pyqtgraph.Qt import QtCore, QtGui

from pyqtgraph.exporters.Exporter import Exporter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtGui, QtCore, QtSvg, USE_PYSIDE
from pyqtgraph import functions as fn
import numpy as np

import subprocess as sp
import glob
import os

import DaXView

import sys,pdb

__all__ = ['MovieExporter']

class MovieExporter(Exporter):
    Name = "Movie File (MP4, AVI)"
    allowCopy = False
    
    def __init__(self, item):
        Exporter.__init__(self, item)
        tr = self.getTargetRect()
        if isinstance(item, QtGui.QGraphicsItem):
            scene = item.scene()
        else:
            scene = item
        bgbrush = scene.views()[0].backgroundBrush()
        bg = bgbrush.color()
        if bgbrush.style() == QtCore.Qt.NoBrush:
            bg.setAlpha(0)

        # Search for DaXView
        wid=self.item.getViewWidget()
        while wid != None and not isinstance(wid,DaXView.DaXView):
            wid = wid.parent()
        self.daxview=wid
            
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'width', 'type': 'int', 'value': tr.width(), 'limits': (0, None)},
            {'name': 'height', 'type': 'int', 'value': tr.height(), 'limits': (0, None)},
            {'name': 'frame rate', 'type': 'int', 'value': 20, 'limits': (1,60)},
            {'name': 'antialias', 'type': 'bool', 'value': True},
            {'name': 'background', 'type': 'color', 'value': bg},
        ])
        self.params.param('width').sigValueChanged.connect(self.widthChanged)
        self.params.param('height').sigValueChanged.connect(self.heightChanged)
        
    def widthChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.height()) / sr.width()
        self.params.param('height').setValue(self.params['width'] * ar, blockSignal=self.heightChanged)
        
    def heightChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.width()) / sr.height()
        self.params.param('width').setValue(self.params['height'] * ar, blockSignal=self.widthChanged)
        
    def parameters(self):
        return self.params
    
    def export(self, fileName=None, toBytes=False, copy=False):
        # Can only export video from DaXView
        if self.daxview == None:
            return
        # Can only write to video
        if toBytes:
            return []
        
        # Get filename from user (the fileSaveDialog function calls export after finishing)
        if fileName is None:
            self.fileSaveDialog(filter=['*.mp4','*.avi'])
            return
        if fileName.endswith('.mp4') == False and fileName.endswith('.avi') == False:
            fileName = fileName + '.mp4'

        # Play entire sequence and store to temporary PNG files
        progress=QtGui.QProgressDialog('Exporting frames...','Abort',
                                       0,int(self.daxview.data.shape[0]*1.1))
        for i in range(self.daxview.data.shape[0]):
            # Set current frame
            self.daxview.setCurrentIndex(i)
            if i>=self.daxview.data.shape[0]:
                progress.setLabelText('Generating video...')
            # Export to png image
            targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
            sourceRect = self.getSourceRect()
            
            #self.png = QtGui.QImage(targetRect.size(), QtGui.QImage.Format_ARGB32)
            #self.png.fill(pyqtgraph.mkColor(self.params['background']))
            w, h = self.params['width'], self.params['height']
            if w == 0 or h == 0:
                raise Exception("Cannot export image with size=0 (requested export size is %dx%d)" % (w,h))
            bg = np.empty((self.params['width'], self.params['height'], 4), dtype=np.ubyte)
            color = self.params['background']
            bg[:,:,0] = color.blue()
            bg[:,:,1] = color.green()
            bg[:,:,2] = color.red()
            bg[:,:,3] = color.alpha()
            self.png = fn.makeQImage(bg, alpha=True)
            
            ## set resolution of image:
            origTargetRect = self.getTargetRect()
            resolutionScale = targetRect.width() / origTargetRect.width()
            
            painter = QtGui.QPainter(self.png)
            #dtr = painter.deviceTransform()
            try:
                self.setExportMode(True, {'antialias': self.params['antialias'], 'background': self.params['background'], 'painter': painter, 'resolutionScale': resolutionScale})
                painter.setRenderHint(QtGui.QPainter.Antialiasing, self.params['antialias'])
                self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
            finally:
                self.setExportMode(False)
            painter.end()
            # Send to png file
            pngFile='temp_'+'{:03d}'.format(i)+'.png'
            self.png.save(pngFile)
            progress.setValue(i)
            if progress.wasCanceled():
                # Clean-up PNGs
                rmfiles=glob.glob('temp_*.png')
                for f in rmfiles:
                    os.remove(f)
                return

        # Convert PNGs to movie
        sp.call(['ffmpeg', '-framerate', str(self.params.param('frame rate').value()),
                           '-y','-loglevel','quiet',
                           '-i', 'temp_%03d.png', 
                           fileName])
        # Clean-up PNGs
        rmfiles=glob.glob('temp_*.png')
        for f in rmfiles:
            os.remove(f)
        progress.close()
    
MovieExporter.register()        
        
