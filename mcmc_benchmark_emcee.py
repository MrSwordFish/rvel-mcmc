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

print ("Starting, time: {t}".format(t=datetime.utcnow()))
true_state = state.State(planets=[{"m":1.2e-3, "a":0.22, "h":0.218, "k":0.015, "l":0.3}, {"m":2.1e-3, "a":0.361, "h":0.16, "k":0.02, "l":2.2}])
obs = observations.FakeObservation(true_state, Npoints=100, error=1e-4, tmax=15.)
#obs = observations.Observation_FromFile(filename='TEST_2-1_COMPACT.vels', Npoints=100)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
ax.plot(*true_state.get_rv_plotting(obs), color="blue")
ax.plot(obs.t, obs.rv, ".r")
ax.set_xticklabels([])
plt.grid()
frame2=fig.add_axes([0.125, -0.17, 0.775, 0.22])        
plt.plot(obs.t,obs.rv-true_state.get_rv(obs.t),'or')
plt.grid()
plt.savefig('emcee_RV_Start.png', bbox_inches='tight')


Nwalkers = 30
ens = mcmc.Ensemble(true_state,obs,scales={"m":1e-3, "a":1., "h":0.2, "k":0.2, "l":np.pi},nwalkers=Nwalkers)
Niter = 1000
chain = np.zeros((Niter,ens.state.Nvars))
chainlogp = np.zeros(Niter)
for i in range(Niter/Nwalkers):
    ens.step_force()
    for j in range(Nwalkers):
        chain[j*Niter/Nwalkers+i] = ens.states[j]
        chainlogp[j*Niter/Nwalkers+i] = ens.lnprob[j]
    print ("Progress: {p:.5}%, time: {t}".format(p=100.*(float(i)/(Niter/Nwalkers)),t=datetime.utcnow()))



fig = plt.figure(figsize=(10,5))
for i in range(ens.state.Nvars):
    ax = plt.subplot(ens.state.Nvars+1,1,1+i)
    ax.set_ylabel(ens.state.get_keys()[i])
    ax.plot(chain[:,i])
ax = plt.subplot(ens.state.Nvars+1,1,ens.state.Nvars+1)
ax.set_ylabel("$\log(p)$")
ax.plot(chainlogp)    
plt.savefig('emcee_Chains.png', bbox_inches='tight')

fig = plt.figure(figsize=(13,5))
ax = plt.subplot(111)
for c in np.random.choice(Niter,100):
    s = ens.state.deepcopy()
    s.set_params(chain[c])
    ax.plot(*s.get_rv_plotting(obs), alpha=0.1, color="gray")
ax.plot(*true_state.get_rv_plotting(obs), color="blue")
ax.errorbar(obs.t, obs.rv, yerr=obs.err, ".r")
ax.set_xticklabels([])
#ax.errorbar(,y,yerr=,fmt='o')
plt.grid()
ax2=fig.add_axes([0.125, -0.63, 0.775, 0.7]) 
plt.plot(*ens.state.get_rv_plotting(obs), alpha=0.8,color="black")
ax2.plot(obs.t, obs.rv, ".r")
ax2.set_xticklabels([])
plt.grid()
ax3=fig.add_axes([0.125, -0.9, 0.775, 0.23])        
plt.plot(obs.t,obs.rv-ens.state.get_rv(obs.t),'or')
plt.grid()

plt.savefig('emcee_RV_trails.png', bbox_inches='tight')

figure = corner.corner(chain, labels=s.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":20})
plt.savefig('emcee_Corners.png', bbox_inches='tight')
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
plt.savefig('emcee_AC_times.png', bbox_inches='tight')
