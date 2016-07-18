import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np
import corner
from datetime import datetime

runName = "_realtest_7__"
logging=True

def AutoCorrelation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result 

def writingToLog(obj, logging):
    if(logging):
        with open("log{r}".format(r=runName),"a") as a:
            for index,value in np.ndenumerate(obj):
                a.write("{v} ".format(v=value))
            a.write("\n")
    else:
        a=None

print ("Starting, run:'{r}', time: {t}".format(t=datetime.utcnow(),r=runName))
writingToLog("START",logging); writingToLog(datetime.utcnow(),logging)
#true_state = state.State(planets=[{"m":1.2e-3, "a":1.42, "h":0.218, "k":0.015, "l":0.1}, {"m":2.1e-3, "a":2.61, "h":0.16, "k":0.02, "l":2.2}])
true_state = state.State(planets=[{"m":0.94e-3, "a":0.226, "h":-0.045, "k":-0.015, "l":1.265}, {"m":1.965e-3, "a":0.307, "h":-0.035, "k":-0.00, "l":1.76}])
#obs = observations.FakeObservation(true_state, Npoints=200, error=1.5e-4, errorVar=2.5e-5, tmax=(30))
obs = observations.Observation_FromFile(filename='TEST_3-2_COMPACT.vels', Npoints=100)
fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.plot(*true_state.get_rv_plotting(obs), color="blue")
plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
ax.set_xticklabels([])
plt.grid()
frame2=fig.add_axes([0.125, -0.17, 0.775, 0.22])        
plt.errorbar(obs.t, true_state.get_rv(obs.t)-obs.rv, yerr=obs.err, fmt='.r')
plt.grid()
plt.savefig('emcee_RV_Start{r}.png'.format(r=runName), bbox_inches='tight')
writingToLog("OBSRV",logging); writingToLog(obs.rv,logging)
writingToLog("STARTSTATE",logging); writingToLog(true_state.get_rv_plotting(obs),logging)
writingToLog("OBSTIMES",logging); writingToLog(obs.rv,logging)

Nwalkers = 24
ens = mcmc.Ensemble(true_state,obs,scales={"m":1.5e-3, "a":0.3, "h":0.1, "k":0.1, "l":np.pi/2.},nwalkers=Nwalkers)
Niter = 10000
chain = np.zeros((Niter,ens.state.Nvars))
chainlogp = np.zeros(Niter)
for i in range(Niter/Nwalkers):
    ens.step_force()
    for j in range(Nwalkers):
        chain[j*Niter/Nwalkers+i] = ens.states[j]
        chainlogp[j*Niter/Nwalkers+i] = ens.lnprob[j]
    if (i%10==1): print ("Progress: {p:.5}%, time: {t}".format(p=100.*(float(i)/(Niter/Nwalkers)),t=datetime.utcnow()))
print("Error(s): {e}".format(e=ens.totalErrorCount))

fig = plt.figure(figsize=(23,12))
for i in range(ens.state.Nvars):
    ax = plt.subplot(ens.state.Nvars+1,1,1+i)
    ax.set_ylabel(ens.state.get_keys()[i])
    ax.plot(chain[:,i])
ax = plt.subplot(ens.state.Nvars+1,1,ens.state.Nvars+1)
ax.set_ylabel("$\log(p)$")
ax.plot(chainlogp)    
plt.savefig('emcee_Chains{r}.png'.format(r=runName), bbox_inches='tight')

fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
averageRandomChain = np.zeros(ens.state.Nvars)
for c in np.random.choice(Niter,45):
    s = ens.state.deepcopy()
    s.set_params(chain[c])
    averageRandomChain += chain[c]
    ax.plot(*s.get_rv_plotting(obs), alpha=0.16, color="darkolivegreen")
    writingToLog("RDMGHOSTS",logging); writingToLog(s.get_rv_plotting(obs),logging)
    writingToLog("RDMGHOSTS LNPROB",logging); writingToLog(s.get_logp(obs),logging)
averageRandomState = ens.state.deepcopy()
averageRandomState.set_params(averageRandomChain/45)
ax.plot(*true_state.get_rv_plotting(obs), color="blue")
plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
ax.set_xticklabels([])
plt.grid()
ax2=fig.add_axes([0.125, -0.63, 0.775, 0.7]) 
plt.plot(*averageRandomState.get_rv_plotting(obs), alpha=0.8,color="black")
print "Average params state (randomly sampled):"
print averageRandomState.get_params()
plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
ax2.set_xticklabels([])
plt.grid()
ax3=fig.add_axes([0.125, -0.9, 0.775, 0.23])        
plt.errorbar(obs.t, averageRandomState.get_rv(obs.t)-obs.rv, yerr=obs.err, fmt='.r')
plt.grid()

plt.savefig('emcee_RV_trails{r}.png'.format(r=runName), bbox_inches='tight')

figure = corner.corner(chain, labels=s.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":20})
plt.savefig('emcee_Corners{r}.png'.format(r=runName), bbox_inches='tight')
fig = plt.figure(figsize=(18,10))
for i in range(s.Nvars):
    fig.suptitle('Autocorelation', fontsize=12)
    ax = plt.subplot(s.Nvars+1,1,1+i)
    ax.set_ylabel(s.get_keys()[i])
    r = AutoCorrelation(chain[:,i])
    ax.plot(r)
    for i in range(len(r)):
        if(r[i] <0.5):
            print "AC time {t}".format(t=i)
            break
plt.savefig('emcee_AC_times{r}.png'.format(r=runName), bbox_inches='tight')
