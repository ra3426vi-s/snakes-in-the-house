import numpy as np
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt

uk = np.random.randint(2, size=100)
#uk=[0,0,0,1,1,0,1,1 ]
jj = 0
ck = []
a = complex()
for e in range(0, 50):
    if uk[jj] == 0 and uk[jj+1] == 0:
            a = complex(1, 1)
    elif uk[jj] == 1 and uk[jj+1] == 0:
            a = complex(-1, 1)
    elif uk[jj] == 0 and uk[jj+1] == 1:
            a = complex(1, -1)
    elif uk[jj] == 1 and uk[jj+1] == 1:
            a = complex(-1, -1)
    if jj == 0:
        ck.append(a)
    else:
        k = jj/2
        ck.append(a)
    jj+=2

in_phase = [x.real for x in ck]
quadrature = [x.imag for x in ck]
plt.figure()
plt.plot(in_phase, quadrature, 'o')
plt.title("Constellation diagram")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)
T = 1  # Symbol duration

t = np.linspace(-15.0, 15.0, int(1.0e4))  # Time interval
g = [(1 if (0 <= t_i <= T) else 0) for t_i in t]  # Rectangular pulse, length T
print(g)
f = fftshift(np.fft.fftfreq(len(t), t[1] - t[0]))
G_f = fftshift(fft(g)) * (T / len(t))

f_theo = np.linspace(-4, 4, int(1.0e3))
x = f_theo * T
G_f_theo = T * np.sinc(x) * np.exp(-1j * x)

plt.rcParams['figure.figsize'] = [12, 6]

plt.figure()

 # Plot the basis function
plt.subplot(1, 2, 1)
plt.plot(t / T, g)
plt.xlim(-4, 4)
plt.grid(True)
plt.xlabel(r'$t/T$', fontsize=14)
plt.ylabel(r'$g_R(t)$', fontsize=14)

# Plot the spectrum of the pulse
plt.subplot(1, 2, 2)
plt.plot(f * T, np.abs(G_f) / np.max(np.abs(G_f)), label=r'$FFT\{g_R(t, T)\}$')
plt.plot(f_theo * T, np.abs(G_f_theo) / T, '--', label=r'$\mathcal{F} \{g_R(t, T)\}$')
plt.xlim(-4, 4)
plt.grid(True)
plt.xlabel(r'$f\cdot T$', fontsize=14)
plt.ylabel(r'$|G_R(f)| / T$', fontsize=14)
plt.legend(prop={'size': 14})


tx=np.convolve(g,ck)
print(tx)
plt.show()


