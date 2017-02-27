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

################################################################################################
#This is a module which facilitates the use of the mcmc classes. 
#As such, it is structured differently.
################################################################################################

class McmcBundle(object):
    def __init__(self, mcmc, chain, chainlogp, clocktimes, obs, Niter, initial_state, is_emcee = False, Nwalkers = 32):
        self.mcmc = mcmc
        self.mcmc_is_emcee = is_emcee
        self.mcmc_Nwalkers = Nwalkers
        self.mcmc_chain = chain
        self.mcmc_chainlogp = chainlogp
        self.mcmc_clocktimes = clocktimes
        self.mcmc_obs = obs
        self.mcmc_Niter = Niter
        self.mcmc_initial_state = initial_state


#Two utility functions meant for the driver class
def AutoCorrelation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result

#Usefull in certain troubleshooting situations.
def WritingToLog(obj, name, logging):
    if(logging):
        with open("log{r}".format(r=nameame),"a") as a:
            for index,value in np.ndenumerate(obj):
                a.write("{v} ".format(v=value))
            a.write("\n")
    else:
        a=None


#functions to be called by notebook/user
def createMH(label, Niter, true_state, obs, scal, step):
    mh = mcmc.Mh(true_state,obs)
    mh.set_scales(scal)
    mh.step_size = step
    chain = np.zeros((0,mh.state.Nvars))
    chainlogp = np.zeros(0)
    tries = 0
    clocktimes = []
    clocktimes.append(datetime.utcnow())
    chainlogp = np.append(chainlogp,true_state.get_logp(obs))
    chain = np.append(chain,[true_state.get_params()],axis=0)
    for i in range(Niter):
        while mh.step()==False:
            chainlogp = np.append(chainlogp,mh.state.get_logp(obs))
            chain = np.append(chain,[mh.state.get_params()],axis=0)
            tries += 1
        tries += 1
        chainlogp = np.append(chainlogp,mh.state.get_logp(obs))
        chain = np.append(chain,[mh.state.get_params()],axis=0)
        if(i % 150 == 1):
            print ("Progress: {p:.5}%, {n} tries have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes.append(datetime.utcnow())
    clocktimes.append(datetime.utcnow())
    print("Acceptance rate: %.3f%%"%(float(Niter)/tries*100))
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(mh, chain, chainlogp, clocktimes, obs, Niter, true_state)
    return bundle, h

def createEns(label, Niter, true_state, obs, Nwalkers, scal):
    ens = mcmc.Ensemble(true_state,obs,scales=scal,nwalkers=Nwalkers)
    listchain = np.zeros((Nwalkers,ens.state.Nvars,0))
    listchainlogp = np.zeros((Nwalkers,0))
    tries=0
    clocktimes = []
    clocktimes.append(datetime.utcnow())
    for i in range(int(Niter/Nwalkers)):
        while ens.step()==False:
            listchainlogp = np.append(listchainlogp, np.reshape(ens.lnprob, (Nwalkers, 1)), axis=1)
            listchain = np.append(listchain, np.reshape(ens.states, (Nwalkers,ens.state.Nvars,1)),axis=2)
            tries += 1
        listchainlogp = np.append(listchainlogp, np.reshape(ens.lnprob, (Nwalkers, 1)), axis=1)
        listchain = np.append(listchain, np.reshape(ens.states, (Nwalkers,ens.state.Nvars,1)),axis=2)
        tries += 1
        if (i%200==1): 
            print ("Progress: {p:.5}%, time: {t}".format(p=100.*(float(i)/(Niter/Nwalkers)),t=datetime.utcnow()))
            clocktimes.append(datetime.utcnow())
    clocktimes.append(datetime.utcnow())
    print("Error(s): {e}".format(e=ens.totalErrorCount))
    print("Acceptance rate: %.3f%%"%(float(Niter/Nwalkers)/tries*100))
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    chain = np.zeros((ens.state.Nvars,0))
    chainlogp = np.zeros(0)
    for i in range(Nwalkers):
        chain = np.append(chain, listchain[i], axis=1)
        chainlogp = np.append(chainlogp, listchainlogp[i])
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(ens, np.transpose(chain), chainlogp, clocktimes, obs, Niter, true_state, is_emcee=True, Nwalkers=Nwalkers)
    return bundle, h

def createSMALA(label, Niter, true_state, obs, eps, alpha, printing_every = 40):
    smala = mcmc.Smala(true_state,obs, eps, alpha)
    smala.print_info_every = printing_every
    chain = np.zeros((0,smala.state.Nvars))
    chainlogp = np.zeros(0)
    tries = 0
    clocktimes = []
    clocktimes.append(datetime.utcnow())
    chainlogp = np.append(chainlogp,true_state.get_logp(obs))
    chain = np.append(chain,[true_state.get_params()],axis=0)
    for i in range(Niter):
        while smala.step()==False:
            chainlogp = np.append(chainlogp,smala.state.get_logp(obs))
            chain = np.append(chain,[smala.state.get_params()],axis=0)
            tries += 1
        tries += 1
        chainlogp = np.append(chainlogp,smala.state.get_logp(obs))
        chain = np.append(chain,[smala.state.get_params()],axis=0)
        if(i % smala.print_info_every == 1):
            print ("Progress: {p:.5}%, {n} tries have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes.append(datetime.utcnow())
    clocktimes.append(datetime.utcnow())
    print("Acceptance rate: %.2f%%"%(float(Niter)/tries*100))
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(smala, chain, chainlogp, clocktimes, obs, Niter, true_state)
    return bundle, h

def preEpsSMALA(true_state, obs, eps, alpha, Niter):
    smala = mcmc.Smala(true_state,obs, eps, alpha)
    print "Trying out eps = {e}".format(e = eps)
    tries = 0
    for i in range(Niter):
        while smala.step()==False:
            tries += 1
        tries += 1
    print "Acc. Rate was {a}".format(a=(float(Niter)/tries))
    if((0.52<=(float(Niter)/tries)) and (0.68>=(float(Niter)/tries))):
        return eps
    elif(0.52>(float(Niter)/tries)):
        mod = 0
        while(mod<=0):
            mod = np.random.normal(loc=0.065, scale=0.025)*8.*np.abs((float(Niter)/tries)-0.6)
        return preEpsSMALA(true_state, obs, eps-mod, alpha, Niter)
    elif(0.68<(float(Niter)/tries)):
        mod = 0
        while(mod<=0):
            mod = np.random.normal(loc=0.065, scale=0.025)*8.*np.abs((float(Niter)/tries)-0.6)
        return preEpsSMALA(true_state, obs, eps+mod, alpha, Niter)

def createALSMALA():
    pass

#Idea: PCGSMALA
def createPCGSMALA():
    pass


def createObs(state, npoint, err, errVar, t):
    obs = observations.FakeObservation(state, Npoints=npoint, error=err, errorVar=errVar, tmax=(t))
    return obs

def ReadObs(filen):
    obs = observations.Observation_FromFile(filename=filen, Npoints=100)
    return obs

def saveObs(obs, true_state, label):
    col1 = obs.t/1.720e-2
    col2 = obs.rv/3.355e-5
    col3 = obs.err/3.355e-5
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    np.savetxt('obs_{ha}.vels'.format(ha=h.hexdigest()), np.c_[col1, col2, col2])

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

def inLinePlotChains(bundle, size, name='Name_left_empty', save=False):
    fig = plt.figure(figsize=(size[0],size[1]))
    mcmc, chain, chainlogp = bundle.mcmc, bundle.mcmc_chain, bundle.mcmc_chainlogp
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
    if(save):
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
        plt.close('all')


def inLinePlotResults(bundle, Ntrails, size, name='Name_left_empty', save=False):
    if(bundle.mcmc_is_emcee):
        pass
    Niter, mcmc, chain, true_state, obs = bundle.mcmc_Niter, bundle.mcmc, bundle.mcmc_chain, bundle.mcmc_initial_state, bundle.mcmc_obs
    fig = plt.figure(figsize=(size[0],size[1]))
    ax = plt.subplot(111)
    averageRandomChain = np.zeros(mcmc.state.Nvars)
    for c in np.random.randint(Niter/5.,Niter,size=Ntrails):
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
    if(save):
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
        plt.close('all')


#disable this function for now since installing corners on scinet is annoying, this is the fastest way to not do this.
def inLinePlotCorners(bundle, name='Name_left_empty', save=False):
    chain, mcmc, true_state = bundle.mcmc_chain, bundle.mcmc, bundle.mcmc_initial_state
    somestate = mcmc.state.deepcopy()
    #figure = corner.corner(chain, labels=somestate.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":33},max_n_ticks=4)
    #plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
    #plt.close('all')
    pass

def inLinePlotAcTimes(bundle, size, name='Name_left_empty', save=False): 
    chain, mcmc = bundle.mcmc_chain, bundle.mcmc
    somestate = mcmc.state.deepcopy()
    actimes = np.zeros(somestate.Nvars)
    fig = plt.figure(figsize=(size[0],size[1]))
    fig.suptitle('Autocorelation', fontsize=12)
    for i in range(somestate.Nvars):
        ax = plt.subplot(somestate.Nvars+1,1,1+i)
        ax.set_ylabel(somestate.get_keys()[i])
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=13)
        ax.locator_params(axis='y', nbins=3)
        if(bundle.mcmc_is_emcee):
            Nwalkers = bundle.mcmc_Nwalkers
            Niter = bundle.mcmc_Niter
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
        else:
            r = AutoCorrelation(chain[:,i])
            ax.plot(r)
            for j in range(len(r)):
                if(r[j] <0.5):
                    actimes[i] = j
                    break
        print "AC time {t}".format(t=actimes[i])
    if(save):
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
        plt.close('all')
    return actimes

def inLinePlotEmceeAcTimes(bundle, size):
    chain, Niter, Nwalkers, mcmc = bundle.mcmc_chain, bundle.mcmc_Niter, bundle.mcmc_Nwalkers, bundle.mcmc
    somestate = mcmc.state.deepcopy()
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

def loadData(name, h):
    return np.load('{n}_{h}.npy'.format(n=name,h=h.hexdigest()))

def saveData(dat, name, h):
    np.save('{n}_{h}'.format(n=name,h=h.hexdigest()), dat)

def saveAuxSmala(h, true_state, label, Niter, eps, alpha):
    with open('aux_{h}'.format(h=h.hexdigest()), "w") as text_file:
        text_file.write('initial = '+ str(true_state.planets) )
        text_file.write("\nlabel, Niter, Eps, Alpha = '{l}', {n}, {e}, {a}".format(l=label, n=Niter, e=eps, a=alpha))

def saveAuxEmcee(h, true_state, label, Niter, Nwalkers, scal):
    with open('aux_{h}'.format(h=h.hexdigest()), "w") as text_file:
        text_file.write('initial = '+ str(true_state.planets) )
        text_file.write("\nlabel, Niter, Nwalkers, Scale = '{l}', {n}, {s}, {t}".format(l=label, n=Niter, s=Nwalkers, t=scal))

def saveAuxMH(h, true_state, label, Niter, scal, step):
    with open('aux_{h}'.format(h=h.hexdigest()), "w") as text_file:
        text_file.write('initial = '+ str(true_state.planets) )
        text_file.write("\nlabel, Niter, Scale, Stepsize = '{l}', {n}, {s}, {t}".format(l=label, n=Niter, s=scal, t=step))
