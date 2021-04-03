# -*- coding: utf-8 -*-
"""
@author: Morgan GRIERE
@classe : 3PSC2
@matiere: Ma322
"""

#-------------------------------------------
#imports
#-------------------------------------------

import math as m
import numpy as np
import matplotlib.pyplot as pp
import scipy.integrate as sc

#-------------------------------------------
#définitions du programme
#-------------------------------------------

def Euler_Exp(f,y0,h):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        y = y + h*f(y,t[n])
    return Ye

def Euler_Imp(f,y0,h):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        y = y + h*f(Ye[n,:],t[n])
    return Ye

def Rung_Kutta_4(f,y0,h):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        k1 = f(y,t[n])
        k2 = f(y+(h/2)*k1,t[n]+(h/2))
        k3 = f(y+(h/2)*k2,t[n]+(h/2))
        k4 = f(y+h*k3,t[n]+h)
        y = y + (h/6)*(k1+2*k2+2*k3+k4)
    return Ye

def Rung_Kutta_2(f,y0,h):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        k1 = y + (h/2)*f(y,t[n])
        k2 = f(k1,t[n]+(h/2))
        y = y + h*k2
    return Ye

def pendule(Y,t):
    g = 9.81
    L = 1
    w = m.sqrt(g/L)
    Yprime = np.array([Y[1],-(w**2)*m.sin(Y[0])])
    return Yprime

def suspension(Y,t):
    M1 = 15;
    M2 = 200;
    C2 = 1200;
    K1 = 50000;
    K2 = 5000;
    f = -1000
    x1_prime = (1/M1)*(C2*Y[3]-C2*Y[2]+K2*Y[1]-(K1+K2)*Y[0])
    x2_prime = (1/M2)*(C2*Y[2]-C2*Y[3]+K2*Y[0]-K2*Y[1]+f)
    Yprime = np.array([Y[2],Y[3],x1_prime,x2_prime])
    return Yprime

#-------------------------------------------
#programme
#-------------------------------------------

A = (m.pi/2)
phi = 0
g = 9.81
L = 1

pas = 0.04
t = np.arange(0,4.01,pas)

theta = [ ]
for i in t:
    theta.append(A*m.cos(m.sqrt(g/L)*i+phi))


Y0 = np.array([m.pi/2,0])
liste_euler = Euler_Exp(pendule,Y0,pas)
liste_RK = Rung_Kutta_4(pendule,Y0,pas)
Yode = sc.odeint(pendule,Y0,t)

liste_RK2 = Rung_Kutta_2(pendule,Y0,pas)

Y0 = np.array([0,0,0,0])
t_suspension = np.arange(0,3.01,0.03)
Yode_suspension = sc.odeint(suspension,Y0,t_suspension)

#-------------------------------------------
#programme test pour la conclusion
#-------------------------------------------

Y0 = np.array([m.pi/2,0])
liste_euler_implicite = Euler_Imp(pendule,Y0,pas)

theta_c = [ ]
for i in t:
    theta_c.append((m.pi/12)*m.cos(m.sqrt(g/L)*i+phi))

Y0_c = np.array([m.pi/12,0])
liste_euler_c = Euler_Exp(pendule,Y0_c,pas)
liste_euler_implicite_c = Euler_Imp(pendule,Y0_c,pas)
liste_RK_c = Rung_Kutta_4(pendule,Y0_c,pas)
Yode_c = sc.odeint(pendule,Y0_c,t)

#-------------------------------------------
#affichage des courbes
#-------------------------------------------

pp.figure(1)
pp.plot(t,theta,label="fonction theta(t)")
pp.plot(t,liste_euler[:,0],label="Euler explicite",linewidth = 1)
pp.plot(t,liste_RK[:,0],label="Runge_Kutta",color ="yellow",linewidth = 4)
pp.plot(t,Yode[:,0],label="Odeint")
pp.title("Tracés des différentes méthodes de résolution d'équation différentielle")
pp.xlabel('t')
pp.ylabel('Y(t)')
pp.legend()
pp.show()

pp.figure(2)
pp.plot(liste_euler[:,0],liste_euler[:,1],label="Euler explicite")
pp.plot(liste_RK[:,0],liste_RK[:,1],label="Runge_Kutta",color ="yellow",linewidth = 4)
pp.plot(Yode[:,0],Yode[:,1],label="Odeint")
pp.title("Portrait de phase ")
pp.xlabel('theta(t)')
pp.ylabel('Y(t)')
pp.legend()
pp.show()

pp.figure(3)
pp.plot(t,theta,label="fonction theta(t)")
pp.plot(t,liste_RK2[:,0],label="Runge_Kutta ordre 2",color ="yellow",linewidth = 4)
pp.plot(t,liste_RK[:,0],label="Runge_Kutta ordre 4",linewidth = 1)
pp.xlabel('t')
pp.ylabel('Y(t)')
pp.title("Comparaison des méthodes Runge_Kutta")
pp.legend()
pp.show()

pp.figure(4)
pp.plot(t_suspension,Yode_suspension[:,0],label="x1(t)")
pp.plot(t_suspension,Yode_suspension[:,1],label="x2(t)")
pp.title("Tracés de x1(t) et x2(t)")
pp.xlabel('t')
pp.ylabel('Y(t)')
pp.legend()
pp.show()

#-------------------------------------------
#affichage des courbes
#-------------------------------------------

pp.figure(5)
pp.plot(t,theta,label="fonction theta(t)")
pp.plot(t,liste_euler[:,0],label="Euler explicite",color = "yellow",linewidth = 4)
pp.plot(t,liste_euler_implicite[:,0],color = "red",label="Euler implicite")
pp.plot(t,liste_RK[:,0],label="Runge_Kutta",color = "orange",linewidth = 4)
pp.plot(t,Yode[:,0],label="Odeint",color ="black")
pp.title("Tracés des différentes méthodes  avec Euler Implicite")
pp.xlabel('t')
pp.ylabel('Y(t)')
pp.legend()
pp.show()

pp.figure(6)
pp.plot(t,theta_c,label="fonction theta(t)",color ="yellow",linewidth = 6)
pp.plot(t,liste_euler_c[:,0],label="Euler explicite",linewidth = 1)
pp.plot(t,liste_RK_c[:,0],label="Runge_Kutta",linewidth = 3)
pp.plot(t,Yode_c[:,0],label="Odeint",linewidth = 1)
pp.title("Tracés des différentes méthodes avec theta(0) = pi/12")
pp.xlabel('t')
pp.ylabel('Y(t)')
pp.legend()
pp.show()

