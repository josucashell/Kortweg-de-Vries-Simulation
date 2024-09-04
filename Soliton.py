#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:53:34 2022

@author: josucashell
"""

import numpy as np
import matplotlib.pyplot as plt
from PDE_solver import RK2, RK4, RK45
import seaborn as sns

def f(u, x):
    n = len(u)
    fu = np.zeros((n))
    h = x[1] - x[0]
    for i in range(n):
        a = (0.25 * (u[(i+1)%n]**2 - u[i-1]**2) / h)
        b = (0.5 * (u[(i+2)%n] - 2*u[(i+1)%n] + 2*u[i-1] - u[i-2]) / h**3)
        fu[i] = -(a + b)
    return fu

def init_cond(x, a, pos):
    x_centre = x[int(len(x)*pos)]
    u0 = 12*a*a/(np.cosh(a*(x-x_centre)))**2
    return u0

#%%

#find values for dt and h so that we get stable solutions (by hand)

T = 1
dt = 0.001
time_steps = np.linspace(0, T, int(round(T/dt))+1)
L = 30
h = 0.1
space_steps = np.linspace(0, L, int(round(L/h))+1)
a = 2

solver = RK4(f)
solver.set_ic(init_cond(space_steps, a, 1/2))
u, x, t, dt = solver.solve(time_steps, space_steps)
solver.show()

#the slider cant be moved continuously or it breaks

#%%

#Euler method stability

N = 150

A = np.zeros((N, N))
for i in range(2, N-2):
    A[i, i+1] = -2
    A[i, i+2] = 1
    A[i, i-1] = 2
    A[i, i-2] = -1
A[0, 1] = -2
A[0, 2] = 1
A[1, 0] = 2
A[1, 2] = -2
A[1, 3] = 1
A[-2, -4] = -1
A[-2, -3] = 2
A[-2, -1] = -2
A[-1, -3] = -1
A[-1, -2] = 2

def method_powers(A):
    e_val0 = 0
    rel_diff = 1
    vn = np.ones(np.size(A, 0))
    count = 0
    while rel_diff > 1e-16:
        vn1 = np.matmul(A, vn)
        e_val1 = (np.matmul(vn, vn1)/np.inner(vn, vn))
        rel_diff = (abs(e_val0-e_val1)/e_val1)
        e_val0 = e_val1.copy()
        vn = vn1.copy()
        count += 1
    return e_val0

def QR_decomp(A):
    N = np.size(A, 0)
    Q = np.zeros((N, N))
    u = np.zeros((N, N))
    for i in range(N):
        proj = 0
        for j in range(i):
            proj += (np.dot(Q[:, j], A[:, i])/np.dot(Q[:, j], Q[:, j]))*Q[:, j]
        u[:, i] = A[:, i] - proj
        Q[:, i] = u[:, i]/np.sqrt(np.dot(u[:, i], u[:, i]))
    R = np.matmul(np.transpose(Q), A)
    return Q, R

def QR_evalue(A):
    N = np.size(A, 0)
    e_val = []
    for i in range(N-1):
        count = 0
        while sum(abs(A[-1, :-1])) > 1e-9:
            Q, R = QR_decomp(A-np.identity(np.size(A, 0))*A[-1, -1])
            Am = np.matmul(R, Q) + np.identity(np.size(A, 0))*A[-1, -1]
            A = Am.copy()
            count += 1
            if count > 1e2:
                break
        e_val.append(A[-1, -1])
        A = A[:-1, :-1].copy()
    e_val.append(A[-1, -1])
    return e_val

def QR_modified_decomp(A):
    Q = np.zeros((np.size(A, 0), np.size(A, 0)))
    Q[:, 0] = A[:, 0]/np.sqrt(np.dot(A[:, 0], A[:, 0]))
    for i in range(1, np.size(A, 0)):
        Q[:, i] = A[:, i]
        for j in range(i):
            Q[:, i] = Q[:, i] - \
                Q[:, j]*(np.dot(Q[:, j], Q[:, i]))
        Q[:, i] = Q[:, i]/np.sqrt(np.dot(Q[:, i], Q[:, i]))
    R = np.matmul(np.transpose(Q), A)
    return Q, R

#this might take a while

b = np.linspace(0, -5, 100)
e_val = []
for i in range(len(b)):
    G = np.identity(N) + b[i]*A
    e = method_powers(G)
    if e < 1:
        e_val.append(QR_evalue(max(G)))
    else:
        e_val.append(e)

print(e_val)

#They all should be grater or equal to 1

#%%

#plots of sech for different values of a (show that it gets wider for small a
# and more peaked for large a)

a = np.linspace(1, 3, 5)
x = np.linspace(-1.5, 1.5, 400)
for i in a:
    u = 12*i*i/(np.cosh(i*(x)))**2
    plt.plot(x, u, label=r"$\alpha$ = {}".format(i))

plt.title("Solutions of KdeV eqn")
plt.grid()
plt.legend()
plt.show()

#%%

#Show that for different a, the wave travels faster (non linearity)
a = np.linspace(1,3,5)
T = 0.75
dt = 0.001
time_steps = np.linspace(0, T, int(round(T/dt))+1)
L = 40
h = 0.1
space_steps = np.linspace(0, L, int(round(L/h))+1)

colors = ["#19126B", "#11379C", "#11937F", "#10DC96", "#96EF0D"]

solver = RK4(f)
for i in range(len(a)):
    solver.set_ic(init_cond(space_steps, a[-(i+1)], 1/5))
    u, x, t, dt = solver.solve(time_steps, space_steps)
    plt.plot(x, u[-1, :], label=r"$\alpha$ = {}".format(a[-(i+1)]), color=colors[-(i+1)])

plt.title("Solutions of KdV eqn at t = 0.75 $s$")
plt.xlabel("Position $x$ $(m)$")
plt.ylabel("$u(x)$ $(m)$")
plt.grid()
plt.legend()
plt.show()
plt.tight_layout()

#%%

#plot of velocity as a function of a

a = np.linspace(1,3,5)
T = 0.1
dt = 0.0001
time_steps = np.linspace(0, T, int(round(T/dt))+1)
L = 50
h = 0.1
space_steps = np.linspace(0, L, int(round(L/h))+1)
x_th = np.linspace(0, 100, 50000)
solver = RK4(f)

v = []
v_err = []
v_theory = []
for i in a:
    u0 = 12*i*i/(np.cosh(i*(space_steps-4*i*i*1)))**2
    u0_1 = 12*i*i/(np.cosh(i*(x_th-4*i*i*1.01)))**2
    u0_0 = 12*i*i/(np.cosh(i*(x_th-4*i*i*1)))**2
    v_theory.append((x_th[np.argmax(u0_1)]-x_th[np.argmax(u0_0)])/0.01)
    solver.set_ic(u0)
    u, x, t, dt = solver.solve(time_steps, space_steps)
    v.append((x[np.argmax(u[-1, :])]-x[np.argmax(u[0, :])])/T)
    v_err.append((h/2)/T)

plt.errorbar(a, v_theory, fmt="x", label="Theoretical velocity", color=colors[2])
plt.errorbar(a, v, yerr=v_err, capsize=1, fmt="x", label="Measured velocity", color=colors[0])
alpha = np.linspace(1,3, 100)
coeffs, cov = np.polyfit(a, v, 2, cov=1)
model = np.poly1d(coeffs)
plt.plot(alpha, model(alpha), label="Best fit", color=colors[1])
plt.ylabel("Velocity $ms^{-1}$")
plt.xlabel(r"$\alpha$")
plt.grid()
plt.legend()
plt.tight_layout()
print(coeffs, np.sqrt(cov))

#%%

#see collisions of similar size
T = 3
dt = 0.0001
time_steps = np.linspace(0, T, int(round(T/dt))+1)
L = 30
h = 0.1
space_steps = np.linspace(0, L, int(round(L/h))+1)

a = 1.2
u0_0 = 12*a*a/(np.cosh(a*(space_steps-4*a*a*0.9)))**2
a = 1
u0_1 = 12*a*a/(np.cosh(a*(space_steps-4*a*a*2.5)))**2
u0 = u0_0 + u0_1

solver = RK4(f)
solver.set_ic(u0)
u, x, t, dt= solver.solve(time_steps, space_steps)
# solver.show()
u_plot = np.zeros(np.size(u, 1))
for i in range(0, np.size(u, 0), 100):
    u_plot = np.vstack([u_plot, u[i, :]])
sns.heatmap(u_plot[::-1, :], cbar_kws={'label': "$u(x)$ $(m)$"}, cmap="mako")
plt.xticks([0, 75, 150, 225, 300], [0, 75, 150, 225, 300], rotation=0)
plt.yticks([0, 75, 150, 225, 300], [3, 2.25, 1.5, 0.75, 0])
plt.ylabel("Time $(s)$")
plt.xlabel("Position $(m)$")
plt.tight_layout()
#%%

#collison of large and small
T = 1.5
dt = 0.0001
time_steps = np.linspace(0, T, int(round(T/dt))+1)
L = 30
h = 0.1
space_steps = np.linspace(0, L, int(round(L/h))+1)

a = 2
u0_0 = 12*a*a/(np.cosh(a*(space_steps-4*a*a*0.2)))**2
a = 1
u0_1 = 12*a*a/(np.cosh(a*(space_steps-4*a*a*3)))**2
u0 = u0_0 + u0_1

solver = RK45(f)
solver.set_ic(u0)
u, x, t, dt= solver.solve(time_steps, space_steps)
# solver.show()
u_plot = np.zeros(np.size(u, 1))
for i in range(0, np.size(u, 0), 5):
    u_plot = np.vstack([u_plot, u[i, :]])
sns.heatmap(u_plot[::-1, :], cbar_kws={'label': "$u(x)$ $(m)$"}, cmap="mako")
plt.xticks([0, 75, 150, 225, 300], [0, 75, 150, 225, 300], rotation=0)
plt.yticks([0, 75, 150, 225, 300], [1.5, 1.125, 0.75, 0.375, 0])
plt.ylabel("Time $(s)$")
plt.xlabel("Position $(m)$")
plt.tight_layout()

#%%
#plot of the used step sizes

plt.plot(dt)

#%%

#area of the solitons should be conserved
def trap_intergration(u, x):
    area = 0.5*(x[1]-x[0])*u[0]
    for i in range(1, len(u)-1):
        area += (x[i+1]-x[i])*u[i]
    area += 0.5*(x[-1]-x[-2])*u[-1]
    return area
        
area = []
for i in range(np.size(u, 0)):
    area.append(trap_intergration(u[i, :], x))

plt.figure(0)
plt.plot(t, area, color=colors[0])
plt.grid()
plt.ylabel("Area $(m^{2})$")
plt.xlabel("Time $(s)$")
plt.tight_layout()

#%%

#moved slider with plotting to solver class and tried to see furier
#decomposition


T = 1500
dt = 0.1
time_steps = np.linspace(0, T, int(round(T/dt))+1)
L = 1000
h = 1
space_steps = np.linspace(0, L, int(round(L/h))+1)

u0 = 1*np.sin(np.pi*space_steps/L)

solver = RK4(f)
solver.set_ic(u0)
u, x, t, dt = solver.solve(time_steps, space_steps)
# solver.show()
u_plot = np.zeros(np.size(u, 1))
for i in range(0, np.size(u, 0), int(15000/(L/h))):
    u_plot = np.vstack([u_plot, u[i, :]])
sns.heatmap(u_plot[::-1, :], cbar_kws={'label': "$u(x)$ $(m)$"}, cmap="mako")
plt.xticks([0, L/(4*h), L/(2*h), L*3/(4*h), L/h], [0, L/(4), L/(2), L*3/(4), L], rotation=0)
plt.yticks([0, L/(4*h), L/(2*h), L*3/(4*h), L/h], [1500, 1125, 750, 375, 0])
plt.ylabel("Time $(s)$")
plt.xlabel("Position $(m)$")
plt.tight_layout()


#%%
#stability map of pde for RK4, arbitrarily changed values of h and dt until i
#got a stable solution

T = 0.1
dt = 0.001
time_steps = np.linspace(0, T, int(round(T/dt))+1)
L = 30
h = 0.1
space_steps = np.linspace(0, L, int(round(L/h))+1)

a = 1
solver = RK4(f)
solver.set_ic(init_cond(space_steps, a, 1/2))
u, x, t, dt = solver.solve(time_steps, space_steps)
solver.show()


#%%

#Remove the dxxx to see that the solutions are no longer stable

def f_no_dxxx(u, x):
    n = len(u)
    fu = np.zeros((n))
    h = x[1] - x[0]
    for i in range(n):
        a = (0.25 * (u[(i+1)%n]**2 - u[i-1]**2) / h)
        b = 0
        fu[i] = -(a + b)
    return fu

T = 0.7
dt = 0.0001
L = 10
h = 0.05
a = 1
time_steps = np.linspace(0, T, int(round(T/dt))+1)
space_steps = np.linspace(0, L, int(round(L/h))+1)
u0 = init_cond(space_steps, a, 1/4)
solver = RK4(f_no_dxxx)
solver.set_ic(u0)
u, x, t, stable = solver.solve(time_steps, space_steps)
# solver.show()
u_plot = np.zeros(np.size(u, 1))
for i in range(0, np.size(u, 0), int(7000/(L/h))):
    u_plot = np.vstack([u_plot, u[i, :]])
sns.heatmap(u_plot[::-1, :], cbar_kws={'label': "$u(x)$ $(m)$"}, cmap="mako")
plt.xticks([0, L/(4*h), L/(2*h), L*3/(4*h), L/h], [0, L/(4), L/(2), L*3/(4), L], rotation=0)
plt.yticks([0, L/(4*h), L/(2*h), L*3/(4*h), L/h], np.round([T, 3*T/4, T/2, T/4, 0], 3))
plt.ylabel("Time $(s)$")
plt.xlabel("Position $(m)$")
plt.title("Shock Wave simulation")
sns.color_palette("mako", as_cmap=True)
plt.tight_layout()

#%%

# Include a diffusive term of form Duxx

def f_diffus(u, x):
    n = len(u)
    fu = np.zeros((n))
    h = x[1] - x[0]
    for i in range(n):
        a = (0.25 * (u[(i+1)%n]**2 - u[i-1]**2) / h)
        b = -1*(u[(i+1)%n] - 2*u[i] + u[i-1])/(h*h)
        fu[i] = -(a + b)
    return fu

T = 3
L = 20
dt = 0.001
h = 0.1
a = 1
time_steps = np.linspace(0, T, int(round(T/dt))+1)
space_steps = np.linspace(0, L, int(round(L/h))+1)
u0 = init_cond(space_steps, a, 1/4)
solver = RK4(f_diffus)
solver.set_ic(u0)
u, x, t, stable = solver.solve(time_steps, space_steps)
# solver.show()
u_plot = np.zeros(np.size(u, 1))
for i in range(0, np.size(u, 0), int((3000)/(L/h))):
    u_plot = np.vstack([u_plot, u[i, :]])
sns.heatmap(u_plot[::-1, :], cbar_kws={'label': "$u(x)$ $(m)$"}, cmap="mako")
plt.xticks([0, L/(4*h), L/(2*h), L*3/(4*h), L/h], [0, L/(4), L/(2), L*3/(4), L], rotation=0)
plt.yticks([0, L/(4*h), L/(2*h), L*3/(4*h), L/h], np.round([T, 3*T/4, T/2, T/4, 0], 3))
plt.ylabel("Time $(s)$")
plt.xlabel("Position $(m)$")
plt.tight_layout()

#%%

#I tried to use interpolation to make the plots smoother, however, couldnt get it to work

#Interpolation

# def slope(x, t, a):
#     b = a*(x-4*a*a*t)
#     f = -24*a*a*a*np.tanh(b)/(np.cosh(b))**2
#     return f

# def interpolation(un, xn, tn, x, a):
#     u = [un[0]]
#     n = len(xn)
#     h = xn[1]-xn[0]
#     for i in range(len(xn)-1):
#         for j in range(len(x)):
#             if x[j] < xn[i+1] and x[j] > xn[i]:
#                 u_inter = un[i] + (x[j]-xn[i])*slope(un[i], tn, a) \
#                     + ((x[j]-xn[i])**2)*(3*(un[(i+1)%n]-un[i]) - h*(slope(un[i+1], tn, a) + 2*slope(un[i], tn, a)))/(h*h) \
#                     + ((x[j]-xn[i])**3)*(-2*(un[(i+1)%n]-un[i]) + h*(slope(un[i+1], tn, a) + slope(un[i], tn, a)))/(h*h*h)
#                 u.append(u_inter)
#         # u.append(un[i])
#     u.append(un[-1])
#     return u

# T = 0.3
# L = 10
# a = 1
# time_steps = np.linspace(0, T, int(round(T/0.0001))+1)
# space_steps = np.linspace(0, L, int(round(L/0.06))+1)
# u0 = init_cond(space_steps, a, 1/2)
# solver = RK4(f)
# solver.set_ic(u0)
# u, x, t, stable = solver.solve(time_steps, space_steps)


# x_extended = np.linspace(0, 10, 2001)
# u_interpolated = interpolation(u[2430, :], x, t[2430], x_extended, 1)
# plt.plot(x_extended, u_interpolated)

        

        
        
        
        
        
        
        
        
