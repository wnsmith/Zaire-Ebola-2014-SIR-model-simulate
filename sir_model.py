import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def SIR_model(y, t, alpha, beta):
    S, I, R = y

    dS_dt = -beta*S*I
    dI_dt = beta*S - alpha*I
    dR_dt = -alpha*I

    return ([dS_dt, dI_dt, dR_dt])


S0 = 0.9
I0 = 0.1
R0 = 0
alpha = 0.1
beta = 0.35

t = np.linspace(0, 100, 10000)

solution = scipy.integrate.odeint(SIR_model, [S0, I0, R0], t, args=(alpha, beta))
solution = np.array(solution)

plt.figure(figsize=[6, 6])
plt.plot(t, solution[:, 0], label="S")
plt.plot(t, solution[:, 1], label="I")
plt.plot(t, solution[:, 2], label="R")
plt.grid()
plt.xlabel("TIME")

