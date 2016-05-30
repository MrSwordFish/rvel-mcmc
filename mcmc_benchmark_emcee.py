import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np
import corner

def AutoCorrelation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result 

true_state = state.State(planets=[{"m":2e-3, "a":1.3, "h":0.35, "k":0.1, "l":0.}])
obs = observations.FakeObservation(true_state, Npoints=100, error=2e-4, tmax=15.)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
ax.plot(*true_state.get_rv_plotting(obs))
ax.plot(obs.t, obs.rv, ".")


Nwalkers = 10
ens = mcmc.Ensemble(true_state,obs,scales={"m":1e-3, "a":1., "h":0.2, "k":0.2, "l":np.pi},nwalkers=Nwalkers)
Niter = 1000
chain = np.zeros((Niter,ens.state.Nvars))
chainlogp = np.zeros(Niter)
for i in range(Niter/Nwalkers):
    ens.step_force()
    for j in range(Nwalkers):
        chain[j*Niter/Nwalkers+i] = ens.states[j]
        chainlogp[j*Niter/Nwalkers+i] = ens.lnprob[j]



fig = plt.figure(figsize=(10,5))
for i in range(ens.state.Nvars):
    ax = plt.subplot(ens.state.Nvars+1,1,1+i)
    ax.set_ylabel(ens.state.get_keys()[i])
    ax.plot(chain[:,i])
ax = plt.subplot(ens.state.Nvars+1,1,ens.state.Nvars+1)
ax.set_ylabel("$\log(p)$")
ax.plot(chainlogp)    

fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
for c in np.random.choice(Niter,100):
    s = ens.state.deepcopy()
    s.set_params(chain[c])
    ax.plot(*s.get_rv_plotting(tmax=15.), alpha=0.1, color="gray")
ax.plot(*true_state.get_rv_plotting(tmax=15.), color="blue")
ax.plot(obs.t, obs.rv, "r.")    

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



plt.show()
