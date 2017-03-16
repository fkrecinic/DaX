# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:33:58 2017

@author: krecinic
"""

import numpy as np
import scipy.special

def erf(x,fitpars):
    A,B,C,D=fitpars[0:4]
    return A+B*(0.5+0.5*scipy.special.erf((x-C)/D))

def erfexp(x,fitpars):
    A,B,C,D,E=fitpars[0:5]
    return erf(x,fitpars[0:4])-B*exponential(x,[0,1,C,E])
     
def exponential(x,fitpars): 
    A,B,C,D=fitpars[0:4]
    res=A+B*(1-np.exp(-(x-C)/D))
    res[(x-C)<0.0]=0.0
    return res

def gaussian(x,fitpars):
    A,B,C,D=fitpars[0:4]
    return A+B*np.exp(-(x-C)**2/(2*D**2))

def linear(x,fitpars):
    A,B=fitpars[0:2]
    return A*x+B

def lsq_erf(fitpars,xdata,ydata):
    return erf(xdata,fitpars)-ydata

def lsq_erfexp(fitpars,xdata,ydata):
    return erfexp(xdata,fitpars)-ydata

def lsq_exponential(fitpars,xdata,ydata): 
    return exponential(xdata,fitpars)-ydata

def lsq_gaussian(fitpars,xdata,ydata):
    return gaussian(xdata,fitpars)-ydata

def lsq_linear(fitpars,xdata,ydata):
    return linear(xdata,fitpars)-ydata
    
funcs={'Linear': linear,
       'Erf': erf,
       'Exponential': exponential,
       'Gaussian': gaussian,
       'Erf+exp': erfexp}
       
lsqfuncs={'Linear': lsq_linear,
           'Erf': lsq_erf,
           'Exponential': lsq_exponential,
           'Gaussian': lsq_gaussian,
           'Erf+exp': lsq_erfexp}

funcstxt={'Linear': 'A*t+B',
          'Erf': 'A+B*(0.5+0.5*erf((t-C)/D))',
          'Exponential': 'A+B*exp(-(t-C)/D)',
          'Gaussian': 'A+B*exp(-(t-C)^2/(2*D^2))',
          'Erf+exp': 'A+B*(0.5+0.5*erf((t-C)/D))-B*(1-np.exp(-(x-E)/F))'}

funcspnum={'Linear': 2,
           'Erf': 4,
           'Exponential': 4,
           'Gaussian': 4,
           'Erf+exp': 5}
