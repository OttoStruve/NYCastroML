"""
The effect of Sampling
----------------------
Figure 10.4

An illustration of the impact of a sampling window function of resulting PSD.
The top-left panel shows a simulated data set with 40 points drawn from the
function y(t|P) = sin(t) (i.e., f = 1/(2pi) ~ 0.16). The sampling is random,
and illustrated by the vertical lines in the bottom-left panel. The PSD of
sampling times, or spectral window, is shown in the bottom-right panel. The
PSD computed for the data set from the top-left panel is shown in the top-right
panel; it is equal to a convolution of the single peak (shaded in gray) with
the window PSD shown in the bottom-right panel (e.g., the peak at f ~ 0.42 in
the top-right panel can be traced to a peak at f ~ 0.26 in the bottom-right
panel).
"""
# Author: Jake VanderPlas

# Edited by: gully
# Date: Friday, June 6, 2014
# Part of #NYCastroML hack day, woo.
# The goal of this hack is simply to explore the windowing effects on the PSD.
#  Bonus goal:  explore how figure figures 10.4 and 10.14 differ.


# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=12, usetex=False)

#------------------------------------------------------------
# Generate the data

#So we're going to make 32768 possible time bins,
# we're only going to pick 40 observations from that
# Then we use the Python lambda function to define the sine wave
Nbins = 2 ** 15 
Nobs = 40
per1=0.3
f = lambda t: 1.0* np.sin(2.0*np.pi * t / per1) 
 #should this be 2 pi or just pi?

t = np.linspace(-100.0, 200, Nbins*1.0) #time ranges from -100 to 200

#dt is just 300/32768
dt = t[1] - t[0]
# close enough...
# print 300.0/32768.0
# 0.0091552734375

y = f(t)

# select observations: pick a sample of 40 random numbers between 0 and 100.
np.random.seed(42)
t_obs = 100 * np.random.random(40)

#Trying to figure out what this D is doing.
#Ah-ha!  This is just a clever whay to make the window.
#Basically these two lines figure out the index closest to each time sample
D = abs(t_obs[:, np.newaxis] - t)
i = np.argmin(D, 1)

#Finally, this makes the binary window function
t_obs = t[i]
y_obs = y[i]
window = np.zeros(Nbins)
window[i] = 1

#------------------------------------------------------------
# Compute PSDs
Nfreq = Nbins / 2

# The infinitesimal frequency is df: 1/(Nbins) * 1/(dt)
# Note that there are half as many frequencies as time samples.
df = 1. / (Nbins * dt)
f = df * np.arange(Nfreq) 
maxf=max(f)

# Only keep the first half of the FFT, since the second half is redundant.
# Question:  Is this always symmetric?  Why bother with the second term of
# Equation 10.6?
PSD_window = abs(np.fft.fft(window)[:Nfreq]) ** 2
PSD_y = abs(np.fft.fft(y)[:Nfreq]) ** 2
PSD_obs = abs(np.fft.fft(y * window)[:Nfreq]) ** 2

# normalize the true PSD so it can be shown in the plot:
# in theory it's a delta function, so normalization is
# arbitrary

# scale PSDs for plotting
# ...Why the choice of 500?  Is this arbitrary?
PSD_window /= 500
PSD_y /= PSD_y.max()
PSD_obs /= 500

#------------------------------------------------------------
# Prepare the figures
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(bottom=0.15, hspace=0.2, wspace=0.25,
                    left=0.12, right=0.95)

# First panel: data vs time
ax = fig.add_subplot(221)
ax.plot(t, y, '-', c='red')
ax.plot(t_obs, y_obs, '.k', ms=4)
ax.text(0.95, 0.93, "Data", ha='right', va='top', transform=ax.transAxes)
ax.set_ylabel('$y(t)$')
ax.set_xlim(0, 100)
ax.set_ylim(-1.5, 1.8)

# Second panel: PSD of data
ax = fig.add_subplot(222)
ax.fill(f, PSD_y, fc='gray', ec='red')
ax.plot(f, PSD_obs, '-', c='black')
ax.text(0.95, 0.93, "Data PSD", ha='right', va='top', transform=ax.transAxes)
ax.set_ylabel('$P(f)$')
ax.set_xlim(0, maxf)
ax.set_ylim(-0.1, 1.1)

# Third panel: window vs time
ax = fig.add_subplot(223)
ax.plot(t, window, '-', c='black')
ax.text(0.95, 0.93, "Window", ha='right', va='top', transform=ax.transAxes)
ax.set_xlabel('$t$')
ax.set_ylabel('$y(t)$')
ax.set_xlim(0, 100)
ax.set_ylim(-0.2, 1.5)

# Fourth panel: PSD of window
ax = fig.add_subplot(224)
ax.plot(f, PSD_window, '-', c='black')
ax.text(0.95, 0.93, "Window PSD", ha='right', va='top', transform=ax.transAxes)
ax.set_xlabel('$f$')
ax.set_ylabel('$P(f)$')
ax.set_xlim(0, maxf)
ax.set_ylim(-0.1, 1.1)

plt.show()
