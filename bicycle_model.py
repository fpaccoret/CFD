# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:47:28 2019

@author: FabPro
"""
import numpy as np
import control as cl
import matplotlib.pyplot as plt

#%% bicycle model try 2
#using lateral dynamics, tire force = c*alpha
#state vector = [y, beta, phi, phidot]
#y=latpos, beta=sideslip, phi=yaw angle, phidot=yawrate
# rdot = (Cr*b-Cf*a)/J*Beta - (Cr*b^2+Cf*a^2)/J/V*r+(Cf*a)/J*delta
# betadot = -(Cr+Cf)/m/V*beta + (Cr*b)

v=30.0 #m/s
cf=350*4.4 #N/deg
cr=250*4.4 #N/deg
m=500.0/2.2 #kg
lf=1.0 #meters
lr=1.0
iz=50.0 #kg*m^2
delta=15/57.3

A=np.array([[0,v,v,0],
            [0,-(cr+cf)/(m*v),0,(cr*lr-cf*lf)/(m*v**2)-1],
            [0,0,0,1],
            [0,(cr*lr-cf*lf)/iz,0,-(cr*lr**2+cf*lf**2)/(iz*v)]])

B = np.array([[0],[cf/m/v],[0],[cf*lf/iz]])

C = np.array([0,0,0,1])

D = np.zeros(1)

byc=cl.StateSpace(A,B*delta,C,D)

t=np.linspace(0,3,1000)
t,y=cl.step_response(byc,t)

plt.figure(1)
plt.subplot(221)
plt.plot(t,y)
plt.grid()
plt.title('step response, yaw rate in rad/s')

C = np.array([0,0,1,0])
byc=cl.StateSpace(A,B*delta,C,D)
t,y=cl.step_response(byc,t)

plt.subplot(222)
plt.plot(t,y)
plt.grid()
plt.title('step response, sideslip angle in rad')

C = np.array([0,0,0,1])
byc=cl.StateSpace(A,B*delta,C,D)
w=np.linspace(0,5*6.28,1000)
byc_tf=cl.ss2tf(byc) #input steer output yaw rate
plt.figure(2)
cl.bode(byc_tf,omega=w,Hz=True,dB=True)





#%% more recent derivation of bike model

from scipy.integrate import odeint

# eqs
# r = ay/v - dbeta/dt
# ay = r*V + dBeta*V
#m*ay = m*r*V + m*dBeta*V
#m*ay = FyF + FyR
#m*ay = Cf*alphaF+Cr*alphaR
#alphaF=delta-beta+r*a/V
#alphaR=-beta+r*b/V

#J*phiddot=FyF*a-FyR*b

J=2000
a=1.5
b=1.5
m=2300
V=20
Cf=67000 #N/rad
Cr=68000
delta=-0.1

def model(z,t):
    
    beta=z[0]
    r=z[1]
    alphaF=delta-beta-r*a/V
    alphaR=-beta+r*b/V
    FyF=Cf*alphaF
    FyR=Cr*alphaR
    ay=FyF/m+FyR/m
    dbeta=ay/V-r
    dr=FyF*a/J-FyR*b/J
    dz=[dbeta,dr]
    return dz

z0=[0.1,0]

t = np.linspace(0,5,1000)
z = odeint(model,z0,t)

plt.plot(t,z[:,0])
plt.plot(t,z[:,1])
plt.grid()
plt.legend(['sideslip','yaw rate'])


