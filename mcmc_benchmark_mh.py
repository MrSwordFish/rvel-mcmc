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

true_state = state.State(planets=[{"m":1e-3, "a":0.225, "h":0.2, "k":0., "l":np.pi/2.}, {"m":2e-3, "a":0.365, "h":0.14, "k":0., "l":np.pi/2.}])
#obs = observations.FakeObservation(true_state, Npoints=100, error=1e-4, tmax=15.)
obs = observations.Observation_FromFile(filename='TEST_2-1_COMPACT.vels', Npoints=100, tmax=15.)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
ax.plot(*true_state.get_rv_plotting(obs))
ax.plot(obs.t, obs.rv, ".")
plt.savefig('mh_RV_Start.png', bbox_inches='tight')

mh = mcmc.Mh(true_state,obs)
mh.set_scales({"m":1e-4, "a":0.005, "h":0.005, "k":0.005, "l":np.pi/15.})
mh.step_size = 4e-2
Niter = 30000
chain = np.zeros((Niter,mh.state.Nvars))
chainlogp = np.zeros(Niter)
tries = 0
for i in range(Niter):
    tries += mh.step_force()
    chain[i] = mh.state.get_params()
    chainlogp[i] = mh.state.logp
    if(i % 250 == 1):
        print ("Progress: {p:.5}%, {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow()))
print("Acceptance rate: %.3f%%"%(float(Niter)/tries*100))

fig = plt.figure(figsize=(10,5))
for i in range(mh.state.Nvars):
    ax = plt.subplot(mh.state.Nvars+1,1,1+i)
    ax.set_ylabel(mh.state.get_keys()[i])
    ax.plot(chain[:,i])
ax = plt.subplot(mh.state.Nvars+1,1,mh.state.Nvars+1)
ax.set_ylabel("$\log(p)$")
ax.plot(chainlogp)    
plt.savefig('mh_Chains.png', bbox_inches='tight')

fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
for c in np.random.choice(Niter,100):
    s = mh.state.deepcopy()
    s.set_params(chain[c])
    ax.plot(*s.get_rv_plotting(obs), alpha=0.1, color="gray")
ax.plot(*true_state.get_rv_plotting(obs), color="blue")
ax.plot(obs.t, obs.rv, "r.")    
plt.savefig('mh_RV_trails.png', bbox_inches='tight')

figure = corner.corner(chain, labels=s.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":20})
plt.savefig('mh_Corners.png', bbox_inches='tight')
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
plt.savefig('mh_AC_times.png', bbox_inches='tight')
