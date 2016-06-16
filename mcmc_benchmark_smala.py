import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np
import corner
from datetime import datetime

def AutoCorrelation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result 

print ("Starting, time: {t}".format(t=datetime.utcnow()))
true_state = state.State(planets=[{"m":0.45e-3, "a":0.223, "h":0.16, "k":-0.02, "l":0.2}, {"m":2e-3, "a":0.3665, "h":0.10, "k":0.09, "l":2.32}])
#true_state = state.State(planets=[{"m":1e-3, "a":1.225, "h":0.7, "k":0., "l":0.0},{"m":2e-3, "a":2.365, "h":0.14, "k":0., "l":0.0}])
#obs = observations.FakeObservation(true_state, Npoints=200, error=2e-4, tmax=46.)
obs = observations.Observation_FromFile(filename='TEST_2-1_COMPACT.vels', Npoints=100)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
ax.plot(*true_state.get_rv_plotting(obs), color="blue")
ax.plot(obs.t, obs.rv, ".r")
ax.set_xticklabels([])
plt.grid()
frame2=fig.add_axes([0.125, -0.17, 0.775, 0.22])        
plt.plot(obs.t,obs.rv-true_state.get_rv(obs.t),'or')
plt.grid()
plt.savefig('smala_RV_Start.png', bbox_inches='tight')

smala = mcmc.Smala(true_state,obs)
Niter = 200
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


fig = plt.figure(figsize=(10,5))
for i in range(smala.state.Nvars):
    ax = plt.subplot(smala.state.Nvars+1,1,1+i)
    ax.set_ylabel(smala.state.get_keys()[i])
    ax.plot(chain[:,i])
ax = plt.subplot(smala.state.Nvars+1,1,smala.state.Nvars+1)
ax.set_ylabel("$\log(p)$")
ax.plot(chainlogp)    
plt.savefig('smala_Chains.png', bbox_inches='tight')

fig = plt.figure(figsize=(13,5))
ax = plt.subplot(111)
for c in np.random.choice(Niter,100):
    s = smala.state.deepcopy()
    s.set_params(chain[c])
    ax.plot(*s.get_rv_plotting(obs), alpha=0.1, color="gray")
ax.plot(*true_state.get_rv_plotting(obs), color="blue")
ax.plot(obs.t, obs.rv, ".r")
ax.set_xticklabels([])
plt.grid()
ax2=fig.add_axes([0.125, -0.63, 0.775, 0.7]) 
plt.plot(*smala.state.get_rv_plotting(obs), alpha=0.8,color="black")
ax2.plot(obs.t, obs.rv, ".r")
ax2.set_xticklabels([])
plt.grid()
ax3=fig.add_axes([0.125, -0.9, 0.775, 0.23])        
plt.plot(obs.t,obs.rv-smala.state.get_rv(obs.t),'or')
plt.grid()

plt.savefig('smala_RV_trails.png', bbox_inches='tight')

figure = corner.corner(chain, labels=s.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":20})

fig = plt.figure(figsize=(14,8))
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
plt.savefig('smala_AC_times.png', bbox_inches='tight')