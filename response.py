import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt, cos

DELAYS = [10.0, 16.18, 24.14, 33.03]
C = 343 # Speed of sound at 20C, in m/s
SIZE = 0.025 # Size unit in metre - pipe lengths are DELAY * SIZE.

def response(r, theta, f):
    """Return response amplitude as function of frequency.

    r is distance from front of driver in m. 
    Theta is angle in radians, 0 is directly in front, pi is directly behind. 
    f is frequency array in Hz
    """
    ampf = 1.0 / r / r  # amplitude factor at front
    hl = [(d* SIZE, sqrt(r * r + d * d * SIZE * SIZE + 2 * cos(theta) * r * d * SIZE)) for d in DELAYS] # list of distances from back ports
    delays = [(h[0] + h[1] - r) / C for h in hl] # propagation time in seconds
    ampb = [1.0 / h[1] / h[1] for h in hl ] # amplitude factors for back ports
    ret = ampb[0] * np.cos(delays[0] * f * 2 *pi)
    for n in range(1, len(DELAYS)) :
        ret = ret +  ampb[n] * np.cos(delays[n] * f * 2 *pi)
    ret = ampf - ret / len(DELAYS)
    return r*r*np.abs(ret)


def polar(r, theta, f):
    """Return response amplitude as function of angle.

    r is distance from front of driver in m. 
    Theta is angle in radians, 0 is directly in front, pi is directly behind. 
    f is frequency array in Hz
    """
    ampf = 1.0 / r / r  # amplitude factor at front
    hl = [(d* SIZE, np.sqrt(r * r + d * d * SIZE * SIZE + 2 * np.cos(theta) * r * d * SIZE)) for d in DELAYS] # list of distances from back ports
    delays = [(h[0] + h[1] - r) / C for h in hl] # propagation time in seconds
    ampb = [1.0 / h[1] / h[1] for h in hl ] # amplitude factors for back ports
    ret = ampb[0] * np.cos(delays[0] * f * 2 *pi)
    for n in range(1, len(DELAYS)) :
        ret = ret +  ampb[n] * np.cos(delays[n] * f * 2 *pi)
    ret = ampf - ret / len(DELAYS)
    return r*r*np.abs(ret)

fr = np.arange(20.0, 400.0, 2.0)
R = 50
lr = 20.0 * np.log10(response(R, 0, fr))
sr = 20.0 * np.log10(response(R, pi/2, fr))
br = 20.0 * np.log10(response(R, pi, fr))


fig = plt.figure(figsize=(5, 10))
ax = plt.subplot(2, 1, 1)

plt.plot(fr, lr, fr, sr, fr, br)
#plt.yscale('log')


# plt.ylim(1e-14, 1e-8)
plt.xlabel("f(Hz)")
plt.ylabel("Response (dB)")

ax = plt.subplot(2, 1, 2, projection='polar')
tr = np.arange(0.0, 2*pi + 0.1, 0.1)
for freq in [25, 50, 100, 200, 400]:
    af = 20 * np.log10(polar(R, tr, freq))
    plt.plot(tr, af)

plt.show()