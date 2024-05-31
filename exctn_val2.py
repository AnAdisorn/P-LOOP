#!/usr/bin/env python
# coding: utf-8

# In[24]:


from scipy import optimize
from scipy.integrate import odeint
from numpy import linalg
from numpy import zeros
from math import pi
from random import seed
from random import random
from math import log
from cmath import exp
import cmath
from math import exp as ex
from math import sqrt
from numpy import empty
import numpy as np
import math
from numpy import array,dot,conjugate
from numpy.linalg import eigh,eig
from matplotlib import pyplot as plt
import sys

Arg=sys.argv[1:]

lamb=int(Arg[0])                               #Sets the iteration until which dynamics is done

N1=300                                       #Sets the total number of iterations to be done

# In[25]:


p=10**(-7)
lamda=lamb
lam=lamda*p                                   #Sets wavelength
c=3.0*10**(8)
h=4.13*10**(16)
freq=c/lam                                    #Defines frequency
freq=freq/h                                   #Converts frequency to atomic units
w=2.0*pi*freq                                 #Sets angular frequency

gamma=0.0024                                    #Dephasing rate with time of 10 femtoseconds



h=4.13*10**(16)
a=2.68                                        #Lattice spacing
g=0.11                                        #Hopping
delta=4.0*pi/(3*sqrt(3)*a)                    #Separation in momentum space of Dirac points
A0=delta                                      #Amplitude
    
T_w= 2.0*pi/w                                 #Cycle duration
T=0.2*T_w
t0=np.linspace(-0.5*T_w,0.5*T_w,300)          #Defining the time array
#t1=np.exp(t0)

print(T)


# In[26]:


def A_field(t):
    
    
    A=-1.0*A0*(np.exp(-1.38*t*t/(T**2)))*(np.cos(w*t))     #Vector potential with Gaussian envelope
    
    return(A)




def E_field(t):                                       #Corresponding Electric field
    
    E1= A0*w*np.exp(-1.38*(t*t)/(T**2))*np.sin(w*t) + A0*np.exp(-1.38*(t*t)/(T**2))*(2.0*1.38*t/(T**2))*np.cos(w*t)
    E=-1.0*E1
    return(E)


# In[27]:


def dipole_element(q_x,q_y):
    #derivative of the valence band
    
    T=-1.0*g*(np.exp(1.0j*a*q_x)+2.0*np.exp(-1.0j*a*q_x/2.0)*np.cos(sqrt(3.0)*a*q_y/2))
    phi=cmath.phase(T)
    
    
    
    
    y1=np.exp(1.0j*phi)
    y2=np.exp(-1.0j*phi)
    
    
    C= np.array([1.0,y2])/sqrt(2.0)         #conduction band
    V= np.array([1.0,-1.0*y2])/sqrt(2.0)     #valence band
    
    #Calculating the xderivative of the valence band
    
    #dx=a*(np.cos(0.5*a*(3*q_x-sqrt(3)*q_y))+np.cos(0.5*a*(3.0*q_x+sqrt(3)*q_y))-2*np.cos(sqrt(3)*a*q_y))/(4*np.cos)
    
    
    a1= (3.0*q_x-sqrt(3.0)*q_y)*0.5*a
    a2= (3.0*q_x+sqrt(3.0)*q_y)*0.5*a
    a3= (sqrt(3.0)*a*q_y)
    
    dx = a*(np.cos(a1)+np.cos(a2)-2*np.cos(a3))/(4*np.cos(a1)+4*np.cos(a2)+4*np.cos(a3)+6)  #delphidelx
    
    
    #Calculating the yderivative of the valence band 
    
    b1=3.0*a*q_x/2
    b2=sqrt(3.0)*a*q_y/2
    
    
    dy= (sqrt(3.0)*a*np.sin(b1)*np.sin(b2))/(2.0*np.cos(b1-b2)+2.0*np.cos(b1+b2)+2.0*np.cos(2*b2)+3.0) #delphidely
    
    
    #getting the dipole matrix elements
    
    dx1= dx*y2*(1.0j)
    dy1= dy*y2*(1.0j)
    
    Dx= np.array([0.0,dx1])/sqrt(2.0) 
    Dy= np.array([0.0,dy1])/sqrt(2.0)
    
    
    Dip_x=np.dot(np.conj(C),Dx)*(1.0j)    #compute dipole element x component
    
    Dip_y=np.dot(np.conj(C),Dy)*(1.0j)    #compute dipole element y component
    
    return(Dip_x,Dip_y)                      


# In[28]:


def curr_elements(q_x,q_y):
    T=-1.0*g*(np.exp(1.0j*a*q_x)+2.0*np.exp(-1.0j*a*q_x/2.0)*np.cos(sqrt(3.0)*a*q_y/2))
    phi=cmath.phase(T)
    
    E1=abs(T)
    E2=-abs(T)
    
    S1= 1.0*g*(2.0*np.exp(-1.0j*a*q_x/2)*np.sin(sqrt(3.0)*a*q_y/2)*sqrt(3.0)*a/2.0)
    S2=np.conj(S1)
    
    #now calculating the ondiagonal elements
    
    Con_del= +0.5*(T*S2 + np.conj(T)*S1)/abs(T)   #delEcdelky
    
    Val_del= -1.0*Con_del                         #delEvdelky
    
    #now calculating the offdiagonal elements
    
    mom_cv= (-1.0j)*(dipole_element(q_x,q_y)[1])*(E1-E2) # delHdelky[cv]
    
    mom_vc= np.conj(mom_cv) #delHdelky[vc]
    
    return(Con_del,Val_del,mom_cv,mom_vc)
  


# In[29]:


def graphene_bloch1(x,t,q_x,q_y):                           #Variables are crystal momentum and the time
    
    #declared constants
    
    #q_x=0.8
    
    #q_y=-0.8
    
    #properly define the vector potential as the definition here
    
    k_y=q_y+A_field(t)
    T=-1.0*g*(np.exp(1.0j*a*q_x)+2.0*np.exp(-1.0j*a*q_x/2.0)*np.cos(sqrt(3.0)*a*k_y/2)) 
    H=np.array([[0.0,T] , [np.conj(T),0.0]]) #Defining the Hamiltonian
    
    E1=abs(T)               #conduction band energy
    E2=-abs(T)              #valence band energy
    
    
    del_E= E1-E2            #band gap 
    
    
    phi=cmath.phase(T)
    
    y1=np.exp(1.0j*phi)           #to determine the elements of the eigenstates( use form {1, +/- e^(i*phi)})
    y2=np.exp(-1.0j*phi)
    
    #define the conduction and valence band eigenvectors:
    
    C1= (np.array([1.0,y2]))/sqrt(2.0)
    V1= (np.array([1.0,-1.0*y2]))/sqrt(2.0)
    
    
    #define the dipole matrix elements (as in the definition:)
    
    
    dipx=dipole_element(q_x,q_y=k_y)[0]         #calculating the dynamic dipole elements 
    dipy=dipole_element(q_x,q_y=k_y)[1]
    
    
    
    
    
    
    #assigning each ODE variable to the vector x:
    c1=x[0] 
    #c2=x[1]     #we do not need this as the ondiagonal elements are real
    c3=x[1] 
    c4=x[2]
    
    
    #defining the real and imaginary parts of density matrix elements
    
    p1=c1            #on-diagonal term for the valence band
    p2=c3+1.0j*c4    #off- diagonal term (rho_cv)
    
    
    
    
    #defining the eqns (hopefully works !!) we just consider E-y for this example
    
    dc1dt=(1.0j*(E_field(t))*(dipy)*p2 + np.conj(1.0j*(E_field(t))*(dipy)*p2)).real #valence band population
    #dc2dt=(1.0j*(E_field(t))*(dipy)*p2 + np.conj(1.0j*(E_field(t))*(dipy)*p2)).imag
    dc3dt=(-1.0*(1.0j*(del_E)+ gamma)*p2 + 1.0j*E_field(t)*(dipy)*(2.0*p1-1.0)).real #rho_cv_real
    dc4dt=(-1.0*(1.0j*(del_E)+ gamma)*p2 + 1.0j*E_field(t)*(dipy)*(2.0*p1-1.0)).imag #rho_cv_imaginary
    
    
    return[dc1dt,dc3dt,dc4dt]


# In[30]:


#lower valleys

t0= np.linspace(-0.5*T_w,0.5*T_w,300)

N=300                               #Sets the iteration until which dynamics is done
#N=300  #sets the time step upto which we propagate
#tx= np.zeros(N,float) # initialising array for the propagation in time domain

#for i in range (0,N):
    #tx[i]=t0[i]                 #the array

P=np.zeros([250,250],float)
P1=np.zeros([250,250],float)
P2=np.zeros([250,250],float)
P3=np.zeros([250,250],float)
l=a
d=4.0*pi/(3.0*l)
d1=-1.0*d
dp1=np.outer(np.linspace(0,d,250),np.ones(250))
#dp2=dp1.copy().T
dp2 = np.outer(np.ones(250),np.linspace(0,d1,250))
#dp2=np.outer(np.ones(60),np.linspace(0,d1,60))

for o in range(0,250):
    print(o)
    q_x=dp1[o,0]
    for g1 in range(0,250):
        
        #print(g)
        
        #print(g)
        q_y=dp2[0,g1]
        vt=sqrt(3)*o
        ft=-1.0*sqrt(3)*o+250*sqrt(3)
        vt1=-1.0*vt
        ft1=-1.0*ft
        
        x = np.array([1.0,0.0,0.0])
        
        if -g1>vt1 and -g1>ft1:
            
            
            
            xq=odeint(graphene_bloch1,x,t0,args=(q_x, q_y)) #solves the ode
            d1=xq[:,0] #rho_vv 
            d2=xq[:,1] #rho_cv_real
            d3=xq[:,2] #rho_cv_imag
            
            d4=d2+1.0j*d3 #rho_cv_total
            d5=np.conj(d4)#rho_vc_total
            
            #Calculating intraband current
            ky=q_y + A_field(t=t0[299]) #gives the electron position at the measured time
            I_inter= curr_elements(q_x=q_x,q_y=ky)[0]*(1.0-d1[299])+curr_elements(q_x=q_x,q_y=ky)[1]*(d1[299])
            
            #Calculating interband current
            I_intra= curr_elements(q_x=q_x,q_y=ky)[2]*(d5[299])+np.conj(curr_elements(q_x=q_x,q_y=ky)[2]*(d5[299]))
            
            #Calculating the total current
            I_tot= I_inter+I_intra
            
            
            P[o,g1]=1.0-d1[299] #excitation
            P1[o,g1]=float(I_inter)   #Intraband current(ondiagonal)
            P2[o,g1]=float(I_intra)     #Interband current(offdiagonal)
            P3[o,g1]=float(I_tot)       #Total current (Intra + Inter)
            
        #print(P[o,g1])
        #print(P[o,g])
        
        
Q4=zeros(1,float)   
H1=zeros(1,float)
H2=zeros(1,float)
H3=zeros(1,float)
Q=zeros(1,float)
        
        
Q4[0]=lamb
H1[0]=np.sum(P1)
H2[0]=np.sum(P2)
H3[0]=np.sum(P3)
Q[0]=np.sum(P)


data=np.asarray(P)
data1=np.asarray(P1)
data2=np.asarray(P2)
data3=np.asarray(P3)

#print(H1,H2,H3)

np.save('exctn2'+str(lamb), data)
np.save('intracurr2'+str(lamb), data1)
np.save('intercurr2'+str(lamb), data2)
np.save('totcurr2'+str(lamb), data3)


Dataout1=np.column_stack((Q4,H1))  #intraband
Dataout2=np.column_stack((Q4,H2))  #interband
Dataout3=np.column_stack((Q4,H3))  #total
Dataout4=np.column_stack((Q4,Q))   #total excitation of conduction

np.savetxt('acurrent2'+str(lamb)+'.txt' ,Dataout1)
np.savetxt('bcurrent2'+str(lamb)+'.txt' ,Dataout2)
np.savetxt('ccurrent2'+str(lamb)+'.txt' ,Dataout3)
np.savetxt('sumextn2'+str(lamb)+'.txt',Dataout4)


# In[26]:


#plt.contourf(dp1,dp2,P,1000,cmap='inferno')
#plt.colorbar()
#plt.show()


# In[ ]:




