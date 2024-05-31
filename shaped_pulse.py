#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#this is the final readymade code for fourier transforms and inverse fourier transforms


# In[1]:


import scipy, matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#The Fourier transform for a Gaussian pulse

def generate_pulse(lamda,phi,N):
    lam=lamda*10**(-7)  #sets wavelength
    c=3*10**8           
    freq=c/lam          #sets frequency
    w=2.0*(np.pi)*freq 
    T_w= 2.0*(np.pi)/w  #sets cycle period
    T=0.2*T_w           #sets pulse width
    
    

    x = np.linspace(-0.5*T_w,0.5*T_w, N, endpoint=False) # N determines the number of sample points
    spacing=(T_w)/N # literally spacing( depends on the initial and final points)
    y = np.exp(-1.38*x*x/(T**2))*(np.cos(w*x+phi)) #phi is an absolute phase
    
    
    y=fft(y)                                      #does the fourier transform (y)
    x=fftfreq(N,spacing)                          # gives the frequencies
    #y = np.fft.fftshift(y)
    #x = np.fft.fftshift(x)
    # 2pi because np.sin takes radians
    
    return(x, y,freq,T_w)


# In[ ]:


#Only while plotting the frequency response or FT, NOT FOR CALCULATING IFFT


y = np.fft.fftshift(y)  # only for plotting
x = np.fft.fftshift(x)  # only for plotting

plt.plot(x,y) #gives Fourier transform


# In[ ]:


# getting the inverse Fourier transform

zf= ifft(y)


# In[ ]:


def array(lamda,N):
    lam=lamda*10**(-7)
    c=3*10**8
    freq=c/lam
    w=2.0*(np.pi)*freq 
    T_w= 2.0*(np.pi)/w 
    X = np.linspace(-0.5*T_w,0.5*T_w, N, endpoint=False)
    spacing=(T_w)/N
    return(X)


# In[ ]:


#Plotting the inverse Fourier transform


X= array(lamda=*,N=*)
plt.plot(X,zf)    #gives inverse Fourier transform


# In[126]:


#Lets define the spectral phase

#w= angular frequency
#w0= central frequency
#width= width of the intensity profile in spectral domain
#a2,a3 are the coefficients in the Taylor series(all lie from -1 to 1)

def spectralphase(w,w0,width,a0,a1,a2,a3,a4):
    z1=(w-w0)/(width)
    phi_spectral=a0+a1*z1+a2*(z1**2)/2+a3*(z1**3)/(3*2)+a4*(z1**4)/(4*3*2)
    return(np.exp(1.0j*phi_spectral))






# In[ ]:





# In[134]:


def gauss(w,w0,width):
    g=2.0*np.log(np.sqrt(2))
    #S fixes the spectral intensity
    z=w-w0
    S=np.exp(-1.0*(g*g*(z)**2)/((width)**2))
    return(np.sqrt(S))

def specpulse(w,w0,width,a0,a1,a2,a3,a4):
    D=gauss(w,w0,width)*spectralphase(w,w0,width,a0,a1,a2,a3,a4)
    return(D)

w0=200
width=0.2
N=6000
spacing=(w0+6*width)/N
w=np.linspace(0,w0+10*width, N, endpoint=False)

L=specpulse(w,w0,width,0.8,1.0,-0.05,0.8,1.0)

L2=specpulse(w,w0,width,0.8,-0.5,-0.05,0.8,1.0)

y=fft(L)
y1=fft(L2)
x=fftfreq(N,spacing) 


# In[137]:


get_ipython().run_line_magic('matplotlib', 'notebook')



y = np.fft.fftshift(y)  # only for plotting
y1= np.fft.fftshift(y1)
x = np.fft.fftshift(x)  # only for plotting

#plt.plot(x,y+np.conj(y))
#plt.plot(x,y1+np.conj(y1))

plt.plot(x,y+np.conj(y)+y1+np.conj(y1))


# In[29]:


from scipy.fft import fft, fftfreq,ifft


# In[ ]:


zr=

