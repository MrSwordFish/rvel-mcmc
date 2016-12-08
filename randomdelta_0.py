import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import driver
import numpy as np
#import corner
import hashlib
from datetime import datetime

niter = 960*2
nitersmala = 30
sigmaMaxRange=40
np.random.seed(2016)
print ("2016")

#Above are the parameters of the run, length of the 2 MCMCs, how many sigmas away, and a seed
scale_of_delta = np.array([(0.00115-0.00075),(0.2286-0.226),(0.04+0.12),(0.10+0.05),(-0.50+1.25),(0.00210-0.00170),(0.3685-0.3650),(0.15+0.15),(0.12+0.12),(2.35-2.00)])/6.
storage = []
for i in range(7,sigmaMaxRange+1):
    q=0.1*i
    Nvec = np.random.randn(10)
    Nvec = np.multiply(Nvec,scale_of_delta)
    initial_state = state.State(planets=[{"m":0.94e-3+q*Nvec[0], "a":0.2275+q*Nvec[1], "h":-0.005+q*Nvec[2], "k":0.03+q*Nvec[3], "l":-1.100+q*Nvec[4]}, {"m":1.965e-3-q*Nvec[5], "a":0.3663-q*Nvec[6], "h":-0.020-q*Nvec[7], "k":0.000-q*Nvec[8], "l":2.15-q*Nvec[9]}])
    obs = driver.ReadObs('TEST_2-1_COMPACT.vels')
    h = hashlib.md5()
    h.update(str(initial_state.planets))
    h.update('label1')
    h2 = hashlib.md5()
    h2.update(str(initial_state.planets))
    h2.update('label2')
    print "Current Sigma Factor = {x}".format(x=q)
    #Setup is done, now run both MCMC
    emcee, chain, chainlogp, clocktimes = driver.createEns('label1', niter, initial_state, obs, 32, {"m":1.5e-3, "a":0.3, "h":0.1, "k":0.1, "l":np.pi/2.})
    smala, chain2, chainlogp2, clocktimes2 = driver.createSMALA('label2', nitersmala, initial_state, obs)
    #Both MCMC have been run for particular 'q', variables returned
    #Process emcee stuff
    driver.inLineSaveChains('chains_emcee_{h}'.format(h=h.hexdigest()),emcee, chain, chainlogp, [20,18])
    s = driver.inLineSaveResults('results_emcee_{h}'.format(h=h.hexdigest()),niter, emcee, chain, initial_state, obs, 50, [20,6])
    driver.inLineSaveCorners('corners_emcee_{h}'.format(h=h.hexdigest()),chain, s, initial_state)
    actimes = driver.inLineSaveEmceeAcTimes('actimes_emcee_{h}'.format(h=h.hexdigest()),chain, niter, 32, [18,6], s)
    eff1 = driver.efficacy(niter, actimes, clocktimes)
    #Process SMALA stuff
    driver.inLineSaveChains('chains_smala_{h2}'.format(h2=h2.hexdigest()),smala, chain2, chainlogp2, [20,12])
    s2 = driver.inLineSaveResults('results_smala_{h2}'.format(h2=h2.hexdigest()),nitersmala, smala, chain2, initial_state, obs, 50, [20,6])
    driver.inLineSaveCorners('corners_smala_{h2}'.format(h2=h2.hexdigest()),chain2, s2, initial_state)
    actimes2 = driver.inLineSaveAcTimes('actimes_smala_{h2}'.format(h2=h2.hexdigest()),chain2, [18,12], s2)
    eff2 = driver.efficacy(nitersmala, actimes2, clocktimes2)
    #Save data for plotting later
    storage.append(np.asarray([Nvec,q,actimes,clocktimes,eff1,actimes2,clocktimes2,eff2]))
    np.save('data_0',storage)

    