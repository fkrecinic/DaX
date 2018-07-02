# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:30:31 2017

@author: krecinic
"""

import os 
import os.path
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from DaXView import DaXView
import MovieExporter
import FitFunctions as ff


import sys,pdb
#QtCore.pyqtRemoveInputHook()
#pdb.set_trace()


# Create main window and qsplitter layout
app = QtGui.QApplication([])
win = QtGui.QMainWindow()
win.resize(1200,800)
splitter = QtGui.QSplitter()
win.setCentralWidget(splitter)

#%%
#==============================================================================
# Create parameter tree 
#==============================================================================
params = [
    {'name': 'Load session', 'type': 'group', 'children': [
        {'name': 'Load', 'type': 'action'},
        {'name': 'Load image data', 'type': 'bool', 'value': True},
    ]},
    {'name': 'Read data sequence', 'type': 'group', 'children': [
        {'name': 'Directory', 'type': 'str', 'value': "/"},
        {'name': 'Select directory', 'type': 'action'},
        {'name': 'Image width', 'type': 'int', 'value': 2048, 'decimals': 6},
        {'name': 'Image height', 'type': 'int', 'value': 2048, 'decimals': 6},
        {'name': 'Binning','type': 'list', 'values': [1,2,4,8], 'value': 1},
        {'name': 'Read sequence', 'type': 'action'}
    ]},
    {'name': 'Spatial axes', 'type': 'group', 'children': [
        {'name': 'Pixel size', 'type': 'float', 'value': 60.23e-3/2048},
        {'name': 'Magnification', 'type': 'float', 'value': 1.0},
        {'name': 'Hide axes', 'type': 'bool', 'value': False}
    ]},
    {'name': 'Time axis', 'type': 'group', 'children': [
        {'name': 'Unit', 'type': 'list', 'values': ['fs','ps','ns','\u00B5s','ms','None']},
        {'name': 'Tbegin', 'type': 'float', 'value': 0.0},
        {'name': 'DeltaT', 'type': 'float', 'value': 10.0},
        {'name': 'Play rate', 'type': 'float', 'value': 20},
        {'name': 'Loop play', 'type': 'bool', 'value': False}
    ]},
    {'name': 'Normalization & differencing', 'type': 'group', 'children': [                                                                           
        {'name': 'Norm. frame intensity', 'type': 'bool', 'value': True},
        {'name': 'Difference image', 'type': 'bool', 'value': False},
        {'name': 'Relative difference', 'type': 'bool', 'value': False},
    ]},
    {'name': 'ROI', 'type': 'group', 'children': [
        {'name': 'Type', 'type': 'list', 'values': ['None','Total','IntX','IntY','Tot. & IntX'], 
                                         'value': 'None'},
        {'name': 'X', 'type': 'int', 'value': 0},
        {'name': 'Y', 'type': 'int', 'value': 0},
        {'name': 'Width', 'type': 'int', 'value': 10},
        {'name': 'Height', 'type': 'int', 'value': 10},
        {'name': 'Angle', 'type': 'int', 'value': 0}
    ]},    
    {'name': 'Fitting functions', 'type': 'group', 'children': [
        {'name': 'Type', 'type': 'list', 'values': ['None','Gaussian',
                    'Erf','Exponential','Linear','Erf+exp'], 'value': 'None'},
        {'name': 'Expression', 'type': 'str', 'value': 'None', 'readonly': True},
        {'name': 'A', 'type': 'float', 'value': 1.0},
        {'name': 'B', 'type': 'float', 'value': 1.0},
        {'name': 'C', 'type': 'float', 'value': 1.0},
        {'name': 'D', 'type': 'float', 'value': 1.0},
        {'name': 'E', 'type': 'float', 'value': 1.0},
#        {'name': 'F', 'type': 'float', 'value': 1.0},
        {'name': 'Set range', 'type': 'bool', 'value': False},
        {'name': 'Fit', 'type': 'action'},
        {'name': 'Fit info', 'type': 'text', 'value': '', 'readonly': True}
    ]},
    {'name': 'Save session', 'type': 'group', 'children': [
        {'name': 'Save', 'type': 'action'},
        {'name': 'Include ROI data', 'type': 'bool', 'value': True},
        {'name': 'Include image data', 'type': 'bool', 'value': False},
        {'name': 'Export video', 'type': 'bool', 'value': False}
    ]}    
]

## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)

def syncToDaxView(imv,p):
    # Sync time axis info from ParameterTree to ImageView
    imv.setTaxis(p.child('Time axis').child('Tbegin').value(), 
                 p.child('Time axis').child('DeltaT').value(),
                 unit=p.child('Time axis').child('Unit').value())
    imv.playspeed=p.child('Time axis').child('Play rate').value()
    # Sync spatial axes
    pa=p.child('Spatial axes')
    imv.setSpaceAxis(pa.child('Pixel size').value()*p.child('Read data sequence').child('Binning').value(),
                     pa.child('Magnification').value())
    # Sync normalization & differencing
    pa=p.child('Normalization & differencing')
    imv.normint=pa.child('Norm. frame intensity').value()
    imv.diffimg=pa.child('Difference image').value()
    imv.normChanged()
    # Sync ROI 
    pa=p.child('ROI')
    imv.roi.setPos((pa.child('X').value(),pa.child('Y').value()),update=False)
    imv.roi.setSize((pa.child('Width').value(),pa.child('Height').value()),
                    update=False)
    imv.roi.setAngle(pa.child('Angle').value())
    imv.roiTypeSelected(pa.child('Type').value())
    # Sync fit function
    pa=p.child('Fitting functions')
    fitpars=np.array([pa.child('A').value(),pa.child('B').value(),
                      pa.child('C').value(),pa.child('D').value()])
    fittype=pa.child('Type').value()
    imv.setFitFunction(fittype,fitpars)

## If anything changes in the tree, print a message
def change(param, changes):
    for param, change, data in changes:
        prt=param.parent()
        
        if prt.name() == 'Load session' and param.name() == 'Load':
            fname=QtGui.QFileDialog.getOpenFileName(filter='Numpy archive (*.npy *.npz)')
            if len(fname)>1:
                fname=fname[0]
            if fname is not '':
                if fname.endswith('.npy'):
                    loaddat=np.load(fname).tolist()
                elif fname.endswith('.npz'):
                    loaddat=np.load(fname)['arr_0'].tolist()
                else:
                    continue
                # Make sure that the load image data setting is NOT restored
                loadimg=p.child('Load session').child('Load image data').value()
                # Restore loaded parameters
                p.restoreState(loaddat['pars'],addChildren=False,removeChildren=False)
                p.child('Load session').child('Load image data').setValue(loadimg)
                # Load image data (if desired by user) 
                if prt.child('Load image data').value():
                    if 'data' in loaddat:
                        imv.data=loaddat['data']
                    else:
                        p.child('Read data sequence').child('Read sequence').activate()
                syncToDaxView(imv,p)

        if param.name() == 'Select directory':
            # Show select directory dialog 
            dirname = str(QtGui.QFileDialog.getExistingDirectory(None, 
                        'Select directory', data))
            if dirname is not '':
                QtCore.QDir.setCurrent(dirname)
            # Write user choice to the string field
            param.parent().child('Directory').setValue(dirname)

        if param.name() == 'Read sequence':
            dirname=prt.child('Directory').value()
            width=prt.child('Image width').value()
            height=prt.child('Image height').value()
            bins=prt.child('Binning').value()
            imv.readSequence(dirname,int(width/bins),int(height/bins))

        if prt.name() == 'Spatial axes' or param.name() == 'Binning':
            pa=p.child('Spatial axes')
            pb=p.child('Read data sequence')
            imv.setSpaceAxis(pa.child('Pixel size').value()*pb.child('Binning').value(),
                             pa.child('Magnification').value(),
                             hide=pa.child('Hide axes').value())

        if prt.name() == 'Time axis':
            imv.setTaxis(prt.child('Tbegin').value(),
                         prt.child('DeltaT').value(),
                         unit=prt.child('Unit').value())
            imv.playspeed=prt.child('Play rate').value()
            imv.looplay=prt.child('Loop play').value()

        if prt.name() == 'Normalization & differencing':
            imv.normint=prt.child('Norm. frame intensity').value()
            imv.diffimg=prt.child('Difference image').value()
            imv.diffimgrel=prt.child('Relative difference').value()
            imv.normChanged()

        if prt.name() == 'ROI':
            if param.name() == 'Type':
                imv.roiTypeSelected(prt.child('Type').value())
            else:
                imv.roi.setPos((prt.child('X').value(),
                                prt.child('Y').value()),update=False)
                imv.roi.setSize((prt.child('Width').value(),
                                 prt.child('Height').value()),update=False)
                imv.roi.setAngle(prt.child('Angle').value())

        if prt.name() == 'Fitting functions':
            fitpars=np.array([prt.child('A').value(),prt.child('B').value(),
                              prt.child('C').value(),prt.child('D').value(),
                              prt.child('E').value()])
            fittype=prt.child('Type').value()
            imv.setFitFunction(fittype,fitpars)
            if param.name() == 'Type':
                if param.value() == 'None':
                    prt.child('Expression').setValue('None')
                else: 
                    prt.child('Expression').setValue(ff.funcstxt[fittype])                
            if param.name() == 'Set range':
                if prt.child('Set range').value():
                    imv.fitRgn.show()
                else:
                    imv.fitRgn.hide()
            if param.name() == 'Fit':
                popt,info=imv.fit(fittype,fitpars)
                prt.child('Fit info').setValue(info)
                with prt.treeChangeBlocker():
                    prt.child('A').setValue(popt[0])
                    prt.child('B').setValue(popt[1])
                    if prt.child('Type').value()!='Linear':
                        prt.child('C').setValue(popt[2])
                        prt.child('D').setValue(popt[3])
                    if prt.child('Type').value()=='Erf+exp':
                        prt.child('E').setValue(popt[4])
#                        prt.child('F').setValue(popt[5])
                imv.setFitFunction(fittype,np.array(popt))
        
        if prt.name() == 'Save session' and param.name() == 'Save':
            savevar={}
            savevar['pars']=p.saveState()
            if prt.child('Include ROI data').value() == True:
                savevar['roidata']=imv.roidata
                savevar['roicoords']=imv.roicoords
            if prt.child('Include image data').value() == True:
                savevar['data']=imv.data            
            fname=QtGui.QFileDialog.getSaveFileName(filter='Numpy archive (*.npy *.npz)')
            if len(fname)>1:
                fname=fname[0]
            if fname is not '':
                print(fname)
                if fname.endswith('.npy'):
                    np.save(fname,savevar)
                else:
                    np.savez_compressed(fname,savevar)

p.sigTreeStateChanged.connect(change)

# Create ParameterTree widget
t = ParameterTree()
t.setParameters(p, showTop=False)

# Add ParameterTree to splitter window
splitter.addWidget(t)     

#%%
#==============================================================================
# Image view
#==============================================================================    
def ROIchange():
    # Get the ROI parameter sub-tree
    pa = p.child('ROI')
    # Read out current ROI positions
    X,Y=imv.roi.pos()
    W,H=imv.roi.size()
    angle=imv.roi.angle()
    # Block treeChange signals (prevents interference between sigRegionChange
    # and sigTreeStateChange)
    with p.treeChangeBlocker():
        # Check if the parameter tree values match the actual ROI position and 
        # update the parameter values ONLY if they dont (otherwise there will be
        # racing between sigTreeStateChange and sigRegionChange)
        if pa.child('X').value() != X:
            pa.child('X').setValue(X)
        if pa.child('Y').value() != Y:
            pa.child('Y').setValue(Y)
        if pa.child('Width').value() != W:
            pa.child('Width').setValue(W)
        if pa.child('Height').value() != H:
            pa.child('Height').setValue(H)
        if pa.child('Angle').value() != angle:
            pa.child('Angle').setValue(imv.roi.angle())

# Add ImageView to the splitter window
imv = DaXView(parent=splitter)
splitter.addWidget(imv)

syncToDaxView(imv,p)
imv.roi.sigRegionChangeFinished.connect(ROIchange)
#%%
#==============================================================================
# Start Qt event loop unless running in interactive mode.
#==============================================================================
win.show()
win.setWindowTitle('Data eXplorer')

if __name__ == '__main__':    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
