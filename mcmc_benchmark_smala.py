import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np
import corner
from datetime import datetime

runName = "_Ben-3-2"
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
#true_state = state.State(planets=[{"m":0.92e-3, "a":0.2275, "h":-0.06, "k":0.015, "l":-1.0}, {"m":1.95e-3, "a":0.3665, "h":0.02, "k":0.0, "l":2.1}])
#true_state = state.State(planets=[{"m":0.92e-3, "a":0.2285, "h":0.05, "k":0.015, "l":-1.8}, {"m":1.95e-3, "a":0.4778, "h":0.01, "k":0.0, "l":2.15}])
#true_state = state.State(planets=[{"m":0.45e-3, "a":0.223, "h":0.16, "k":-0.02, "l":0.2}, {"m":2e-3, "a":0.3665, "h":0.10, "k":0.09, "l":2.32}])
true_state = state.State(planets=[{"m":0.9e-3, "a":0.226, "h":-0.06, "k":-0.015, "l":1.3}, {"m":1.85e-3, "a":0.3057, "h":-0.03, "k":-0.01, "l":1.75}])
#true_state = state.State(planets=[{"m":1e-3, "a":1.225, "h":0.7, "k":0., "l":0.0},{"m":2e-3, "a":2.365, "h":0.14, "k":0., "l":0.0}])
#obs = observations.FakeObservation(true_state, Npoints=60, error=1.5e-4, errorVar=2.5e-5, tmax=(30))
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
plt.savefig('smala_RV_Start{r}.png'.format(r=runName), bbox_inches='tight')
writingToLog("STARTSTATE",logging); writingToLog(true_state.get_rv_plotting(obs),logging)
writingToLog("OBSRV",logging); writingToLog(obs.rv,logging)
writingToLog("OBSTIMES",logging); writingToLog(obs.rv,logging)

smala = mcmc.Smala(true_state,obs)
Niter = 4200
chain = np.zeros((Niter,smala.state.Nvars))
chainlogp = np.zeros(Niter)
tries = 0
for i in range(Niter):
    tries += smala.step_force()
    chain[i] = smala.state.get_params()
    chainlogp[i] = smala.state.logp
    if(i % 5 == 1):
        print ("Progress: {p:.5}%, {n} tries have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
print("Acceptance rate: %.2f%%"%(float(Niter)/tries*100))


fig = plt.figure(figsize=(23,12))
for i in range(smala.state.Nvars):
    ax = plt.subplot(smala.state.Nvars+1,1,1+i)
    ax.set_ylabel(smala.state.get_keys()[i])
    ax.plot(chain[:,i])
ax = plt.subplot(smala.state.Nvars+1,1,smala.state.Nvars+1)
ax.set_ylabel("$\log(p)$")
ax.plot(chainlogp)    
plt.savefig('smala_Chains{r}.png'.format(r=runName), bbox_inches='tight')

fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
averageRandomChain = np.zeros(smala.state.Nvars)
for c in np.random.randint(Niter/4.,Niter,size=45):
    s = smala.state.deepcopy()
    s.set_params(chain[c])
    averageRandomChain += chain[c]
    ax.plot(*s.get_rv_plotting(obs), alpha=0.16, color="darkolivegreen")
    writingToLog("RDMGHOSTS",logging); writingToLog(s.get_rv_plotting(obs),logging)
averageRandomState = smala.state.deepcopy()
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

plt.savefig('smala_RV_trails{r}.png'.format(r=runName), bbox_inches='tight')

figure = corner.corner(chain, labels=s.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":20})
plt.savefig('smala_Corners{r}.png'.format(r=runName), bbox_inches='tight')

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
plt.savefig('smala_AC_times{r}.png'.format(r=runName), bbox_inches='tight')