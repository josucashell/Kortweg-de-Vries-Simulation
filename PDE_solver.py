#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:32:45 2022

@author: josucashell
"""
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

class PDESolver:
    def __init__(self, f):
        self.f = f
        
    def set_ic(self, u0):
        self.u0 = np.array(u0)
        
    def solve(self, time, space):
        self.t = np.array(time)
        n = np.size(self.t)
        self.dt = np.zeros((n))
        self.dt[0] = self.t[1] - self.t[0]
        
        self.x = np.array(space)
        
        self.u = np.zeros((n, np.size(self.x)))
        self.u[0, :] = self.u0
        u_max = max(self.u0)

        for i in range(n-1):
            self.i = i
            self.u[i+1, :], self.dt[i+1] = self.advance()
            self.t[i+1] = self.dt[i+1]+self.t[i]
            if self.t[i+1] > self.t[-1]:
                self.t = np.delete(self.t, np.s_[i+1:])
                self.dt = np.delete(self.dt, np.s_[i+1:])
                self.u = np.delete(self.u, np.s_[i+1:], 0)
                break
            ui_max = np.max(self.u[i+1, :])
            diff = abs(u_max - ui_max)
            if diff > 1000:
                break
            
        return self.u, self.x, self.t, self.dt

    
    def advance(self):
        raise NotImplementedError
    
    def show(self):
        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.u[1, :], lw=2)
        fig.subplots_adjust(left=0.25, bottom=0.25)
        axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        time_slider = Slider(ax=axtime,label='Time', valmin=1, \
            valmax=np.size(self.t)-1, valinit=1, valfmt='%0.0f')
        
        def update(val):
            line.set_ydata(self.u[int(time_slider.val), :])
            fig.canvas.draw_idle()
        
        time_slider.on_changed(update)
        return time_slider
        
        
class ForwardEuler(PDESolver):
    def advance(self):
        u, f, i, t, x = self.u, self.f, self.i, self.t, self.x
        dt = t[i+1] - t[i]
        return u[i, :] + dt * f(u[i, :], x), dt
    
class RK2(PDESolver):
    def advance(self):
        u, f, i, t, x = self.u, self.f, self.i, self.t, self.x
        dt = t[i+1]-t[i]
        k1 = f(u[i, :], x)
        k2 = f(u[i, :] + (dt/2) * k1, x)
        return u[i, :] + dt * k2, dt
    
class RK4(PDESolver):
    def advance(self):
        u, f, i, t, x = self.u, self.f, self.i, self.t, self.x
        dt = t[i+1]-t[i]
        k1 = f(u[i, :], x)
        k2 = f(u[i, :] + (dt/2) * k1, x)
        k3 = f(u[i, :] + (dt/2) * k2, x)
        k4 = f(u[i, :] + dt * k3, x)
        return u[i, :] + (1/6)*dt*(k1 + 2*k2 + 2*k3 + k4), dt

class RK45(PDESolver):
    def advance(self):
        u, f, i, x, dt = self.u, self.f, self.i, self.x, self.dt[self.i]
        k1 = dt*f(u[i, :], x)
        k2 = dt*f(u[i, :] + (1/4) * k1, x)
        k3 = dt*f(u[i, :] + (3/32)*k1 + (9/32)*k2, x)
        k4 = dt*f(u[i, :] + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3, x)
        k5 = dt*f(u[i, :] + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4105)*k4, x)
        k6 = dt*f(u[i, :] - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5, x)
        y = u[i, :] + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 - (1/5)*k5
        z = u[i, :] + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
        n = np.argmax(y)
        s = np.power((11e-2*dt/(2*abs(z[n] - y[n]))), 0.25)
        if s*dt > 1.3e-3:
            return z, 1.3e-3
        elif s*dt < 1e-6:
            return z, 1e-6
        else:
            return z, s*dt

#%%

