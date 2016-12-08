import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np
#corner plots temporarily taken out while running on cluster
#import corner
import hashlib
from datetime import datetime

#Two utility functions meant fro the driver class
def AutoCorrelation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result

def WritingToLog(obj, name, logging):
    if(logging):
        with open("log{r}".format(r=nameame),"a") as a:
            for index,value in np.ndenumerate(obj):
                a.write("{v} ".format(v=value))
            a.write("\n")
    else:
        a=None

#functions to be called by notebook/user
def CreateObs(state, npoint, err, errVar, t):
    obs = observations.FakeObservation(state, Npoints=npoint, error=err, errorVar=errVar, tmax=(t))
    return obs

def ReadObs(filen):
    obs = observations.Observation_FromFile(filename=filen, Npoints=100)
    return obs

def inLinePlotObs(true_state, obs, size):
    fig = plt.figure(figsize=(size[0],size[1]))
    ax = plt.subplot(111)
    ax.plot(*true_state.get_rv_plotting(obs), color="blue")
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax.set_xticklabels([])
    ax.set_ylabel("$Initial RV$", fontsize=28)
    ax.tick_params(axis='both', labelsize=18)
    plt.grid()
    frame2=fig.add_axes([0.125, -0.17, 0.775, 0.22])
    plt.tick_params(axis='both', labelsize=18)  
    frame2.set_ylabel("$Res. RV$", fontsize=28)
    frame2.set_xlabel("$Time$", fontsize=28)      
    plt.errorbar(obs.t, true_state.get_rv(obs.t)-obs.rv, yerr=obs.err, fmt='.r')
    plt.grid()

def createMH(label, Niter, true_state, obs, scal, step):
    mh = mcmc.Mh(true_state,obs)
    mh.set_scales(scal)
    mh.step_size = step
    chain = np.zeros((Niter,mh.state.Nvars))
    chainlogp = np.zeros(Niter)
    tries = 0
    clocktimes = []
    clocktimes.append(datetime.utcnow())
    for i in range(Niter):
        tries += mh.step_force()
        chain[i] = mh.state.get_params()
        chainlogp[i] = mh.state.logp
        if(i % 150 == 1):
            print ("Progress: {p:.5}%, {n} tries have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes.append(datetime.utcnow())
    clocktimes.append(datetime.utcnow())
    print("Acceptance rate: %.3f%%"%(float(Niter)/tries*100))
    print("Saving data to disk")
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    print (h.hexdigest())
    with open('aux_{h}'.format(h=h.hexdigest()), "w") as text_file:
        text_file.write(str(true_state.planets))
        text_file.write("\n {l}, Niter={n}, Scale={s}, Stepsize={t}".format(l=label, n=Niter, s=scal, t=step))
    np.save('chain_MH_{h}'.format(h=h.hexdigest()), chain)
    np.save('chainlogp_MH_{h}'.format(h=h.hexdigest()), chainlogp)
    np.save('clocktimes_MH_{h}'.format(h=h.hexdigest()), clocktimes)
    return mh, chain, chainlogp, clocktimes

def createEns(label, Niter, true_state, obs, Nwalkers, scal):
    ens = mcmc.Ensemble(true_state,obs,scales=scal,nwalkers=Nwalkers)
    chain = np.zeros((Niter,ens.state.Nvars))
    chainlogp = np.zeros(Niter)
    clocktimes = []
    clocktimes.append(datetime.utcnow())
    for i in range(Niter/Nwalkers):
        ens.step_force()
        for j in range(Nwalkers):
            chain[j*Niter/Nwalkers+i] = ens.states[j]
            chainlogp[j*Niter/Nwalkers+i] = ens.lnprob[j]
        if (i%200==1): 
            print ("Progress: {p:.5}%, time: {t}".format(p=100.*(float(i)/(Niter/Nwalkers)),t=datetime.utcnow()))
            clocktimes.append(datetime.utcnow())
    clocktimes.append(datetime.utcnow())
    print("Error(s): {e}".format(e=ens.totalErrorCount))
    print "In total, there has been {n} collisions.".format(n=ens.state.coCount)
    print("Saving data to disk")
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    print (h.hexdigest())
    with open('aux_{h}'.format(h=h.hexdigest()), "w") as text_file:
        text_file.write(str(true_state.planets))
        text_file.write("\n {l}, Niter={n}, Nwalkers={s}, Scale={t}".format(l=label, n=Niter, s=Nwalkers, t=scal))
    np.save('chain_emcee_{h}'.format(h=h.hexdigest()), chain)
    np.save('chainlogp_emcee_{h}'.format(h=h.hexdigest()), chainlogp)
    np.save('clocktimes_emcee_{h}'.format(h=h.hexdigest()), clocktimes)
    return ens, chain, chainlogp, clocktimes

def createSMALA(label, Niter, true_state, obs):
    smala = mcmc.Smala(true_state,obs)
    chain = np.zeros((Niter,smala.state.Nvars))
    chainlogp = np.zeros(Niter)
    tries = 0
    clocktimes = []
    clocktimes.append(datetime.utcnow())
    for i in range(Niter):
        tries += smala.step_force()
        chain[i] = smala.state.get_params()
        chainlogp[i] = smala.state.logp
        if(i % 40 == 1):
            print ("Progress: {p:.5}%, {n} tries have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes.append(datetime.utcnow())
    clocktimes.append(datetime.utcnow())
    print("Acceptance rate: %.2f%%"%(float(Niter)/tries*100))
    print "In total, there has been {n} collisions.".format(n=smala.state.coCount)
    print("Saving data to disk")
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    print (h.hexdigest())
    with open('aux_{h}'.format(h=h.hexdigest()), "w") as text_file:
        text_file.write(str(true_state.planets))
        text_file.write("\n {l}, Niter={n}".format(l=label, n=Niter))
    np.save('chain_smala_{h}'.format(h=h.hexdigest()), chain)
    np.save('chainlogp_smala_{h}'.format(h=h.hexdigest()), chainlogp)
    np.save('clocktimes_smala_{h}'.format(h=h.hexdigest()), clocktimes)
    return smala, chain, chainlogp, clocktimes

def inLinePlotChains(mcmc, chain, chainlogp, size):
    fig = plt.figure(figsize=(size[0],size[1]))
    for i in range(mcmc.state.Nvars):
        ax = plt.subplot(mcmc.state.Nvars+1,1,1+i)
        ax.set_ylabel(mcmc.state.get_keys()[i])
        ax.tick_params(axis='x', labelbottom='off')
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=18)
        ax.locator_params(axis='y', nbins=3)
        ax.plot(chain[:,i])
    ax = plt.subplot(mcmc.state.Nvars+1,1,mcmc.state.Nvars+1)
    ax.set_ylabel("$\log(p)$")
    ax.yaxis.label.set_size(28)
    ax.tick_params(axis='both', labelsize=18)
    ax.locator_params(axis='y', nbins=3)
    ax.plot(chainlogp)    

def inLinePlotResults(Niter, mcmc, chain, true_state, obs, Ntrails, size):
    fig = plt.figure(figsize=(size[0],size[1]))
    ax = plt.subplot(111)
    averageRandomChain = np.zeros(mcmc.state.Nvars)
    for c in np.random.randint(Niter/4.,Niter,size=Ntrails):
        s = mcmc.state.deepcopy()
        s.set_params(chain[c])
        averageRandomChain += chain[c]
        ax.plot(*s.get_rv_plotting(obs), alpha=0.12, color="darkolivegreen")
    averageRandomState = mcmc.state.deepcopy()
    averageRandomState.set_params(averageRandomChain/Ntrails)
    ax.plot(*true_state.get_rv_plotting(obs), color="blue")
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax.set_xticklabels([])
    plt.grid()
    ax2=fig.add_axes([0.125, -0.63, 0.775, 0.7]) 
    ax.set_ylabel("Initial RV")
    ax2.set_ylabel("Average Result RV")
    ax.yaxis.label.set_size(28)
    ax2.yaxis.label.set_size(28)
    ax.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    plt.plot(*averageRandomState.get_rv_plotting(obs), alpha=0.8,color="black")
    print "Resulting average params state (randomly sampledriver.ind):"
    print averageRandomState.get_keys()
    print averageRandomState.get_params()
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax2.set_xticklabels([])
    plt.grid()
    ax3=fig.add_axes([0.125, -0.9, 0.775, 0.23])  
    ax3.tick_params(axis='both', labelsize=18)  
    ax3.yaxis.label.set_size(28)
    ax3.set_ylabel("Residual RV")    
    plt.errorbar(obs.t, averageRandomState.get_rv(obs.t)-obs.rv, yerr=obs.err, fmt='.r')
    plt.grid()
    return s

#disable this function for now since installing corners on scinet is annoying, this is the fastest way to not do this.
def inLinePlotCorners(chain, somestate, true_state):
#    figure = corner.corner(chain, labels=somestate.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":33},max_n_ticks=4)
    pass

def inLinePlotAcTimes(chain, size, somestate):
    actimes = np.zeros(somestate.Nvars)
    fig = plt.figure(figsize=(size[0],size[1]))
    for i in range(somestate.Nvars):
        fig.suptitle('Autocorelation', fontsize=12)
        ax = plt.subplot(somestate.Nvars+1,1,1+i)
        ax.set_ylabel(somestate.get_keys()[i])
        r = AutoCorrelation(chain[:,i])
        ax.plot(r)
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=13)
        ax.locator_params(axis='y', nbins=3)
        for j in range(len(r)):
            if(r[j] <0.5):
                print "AC time {t}".format(t=j)
                actimes[i] = j
                break
    return actimes

def inLinePlotEmceeAcTimes(chain, Niter, Nwalkers, size, somestate):
    actimes = np.zeros(somestate.Nvars)
    fig = plt.figure(figsize=(size[0],size[1]))
    fig.suptitle('Autocorelation', fontsize=12)
    for i in range(somestate.Nvars):
        ax = plt.subplot(somestate.Nvars+1,1,1+i)
        ax.set_ylabel(somestate.get_keys()[i])
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=13)
        ax.locator_params(axis='y', nbins=3)
        temp = np.zeros(Niter/Nwalkers)
        x = 0
        for k in range(Nwalkers):
            for p in range(Niter/Nwalkers):
                temp[p] = chain[(Niter/Nwalkers)*k+p,i]
            y = AutoCorrelation(temp)
            ax.plot(y, alpha=0.18, color="darkolivegreen")
            for j in range(len(y)):
                if(y[j] <0.5):
                    actimes[i] += j
                    break
        actimes[i] /= Nwalkers
        print "AC time {t}".format(t=actimes[i])
    return actimes

def efficacy(Niter, AC, clockTimes):
    dt = (clockTimes[len(clockTimes)-1]-clockTimes[1]).total_seconds()
    return (Niter/(dt*np.amax(AC)))

#
#Alternate version of the functions above except the plots are SAVED to the disk in ./mcmcplots.
#
#

def inLineSaveChains(name, mcmc, chain, chainlogp, size):
    fig = plt.figure(figsize=(size[0],size[1]))
    for i in range(mcmc.state.Nvars):
        ax = plt.subplot(mcmc.state.Nvars+1,1,1+i)
        ax.set_ylabel(mcmc.state.get_keys()[i])
        ax.tick_params(axis='x', labelbottom='off')
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=18)
        ax.locator_params(axis='y', nbins=3)
        ax.plot(chain[:,i])
    ax = plt.subplot(mcmc.state.Nvars+1,1,mcmc.state.Nvars+1)
    ax.set_ylabel("$\log(p)$")
    ax.yaxis.label.set_size(28)
    ax.tick_params(axis='both', labelsize=18)
    ax.locator_params(axis='y', nbins=3)
    ax.plot(chainlogp) 
    plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
    plt.close('all')

def inLineSaveResults(name, Niter, mcmc, chain, true_state, obs, Ntrails, size):
    fig = plt.figure(figsize=(size[0],size[1]))
    ax = plt.subplot(111)
    averageRandomChain = np.zeros(mcmc.state.Nvars)
    for c in np.random.randint(Niter/4.,Niter,size=Ntrails):
        s = mcmc.state.deepcopy()
        s.set_params(chain[c])
        averageRandomChain += chain[c]
        ax.plot(*s.get_rv_plotting(obs), alpha=0.12, color="darkolivegreen")
    averageRandomState = mcmc.state.deepcopy()
    averageRandomState.set_params(averageRandomChain/Ntrails)
    ax.plot(*true_state.get_rv_plotting(obs), color="blue")
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax.set_xticklabels([])
    plt.grid()
    ax2=fig.add_axes([0.125, -0.63, 0.775, 0.7]) 
    ax.set_ylabel("Initial RV")
    ax2.set_ylabel("Average Result RV")
    ax.yaxis.label.set_size(28)
    ax2.yaxis.label.set_size(28)
    ax.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    plt.plot(*averageRandomState.get_rv_plotting(obs), alpha=0.8,color="black")
    print "Resulting average params state (randomly sampledriver.ind):"
    print averageRandomState.get_keys()
    print averageRandomState.get_params()
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax2.set_xticklabels([])
    plt.grid()
    ax3=fig.add_axes([0.125, -0.9, 0.775, 0.23])  
    ax3.tick_params(axis='both', labelsize=18)  
    ax3.yaxis.label.set_size(28)
    ax3.set_ylabel("Residual RV")    
    plt.errorbar(obs.t, averageRandomState.get_rv(obs.t)-obs.rv, yerr=obs.err, fmt='.r')
    plt.grid()
    plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
    plt.close('all')
    return s

#disable this function for now since installing corners on scinet is annoying, this is the fastest way to not do this.
def inLineSaveCorners(name, chain, somestate, true_state):
    #figure = corner.corner(chain, labels=somestate.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":33},max_n_ticks=4)
    #plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
    #plt.close('all')
    pass

def inLineSaveAcTimes(name, chain, size, somestate):
    actimes = np.zeros(somestate.Nvars)
    fig = plt.figure(figsize=(size[0],size[1]))
    for i in range(somestate.Nvars):
        fig.suptitle('Autocorelation', fontsize=12)
        ax = plt.subplot(somestate.Nvars+1,1,1+i)
        ax.set_ylabel(somestate.get_keys()[i])
        r = AutoCorrelation(chain[:,i])
        ax.plot(r)
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=13)
        ax.locator_params(axis='y', nbins=3)
        for j in range(len(r)):
            if(r[j] <0.5):
                print "AC time {t}".format(t=j)
                actimes[i] = j
                break
    plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
    plt.close('all')
    return actimes

def inLineSaveEmceeAcTimes(name, chain, Niter, Nwalkers, size, somestate):
    actimes = np.zeros(somestate.Nvars)
    fig = plt.figure(figsize=(size[0],size[1]))
    fig.suptitle('Autocorelation', fontsize=12)
    for i in range(somestate.Nvars):
        ax = plt.subplot(somestate.Nvars+1,1,1+i)
        ax.set_ylabel(somestate.get_keys()[i])
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=13)
        ax.locator_params(axis='y', nbins=3)
        temp = np.zeros(Niter/Nwalkers)
        x = 0
        for k in range(Nwalkers):
            for p in range(Niter/Nwalkers):
                temp[p] = chain[(Niter/Nwalkers)*k+p,i]
            y = AutoCorrelation(temp)
            ax.plot(y, alpha=0.18, color="darkolivegreen")
            for j in range(len(y)):
                if(y[j] <0.5):
                    actimes[i] += j
                    break
        actimes[i] /= Nwalkers
        print "AC time {t}".format(t=actimes[i])
    plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
    plt.close('all')
    return actimes
