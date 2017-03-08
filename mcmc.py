import numpy as np
import emcee
from scipy import stats
import rebound
from datetime import datetime
import sys
import traceback

'''
Parent MCMC class
'''
class Mcmc(object):
    def __init__(self, initial_state, obs):
        self.state = initial_state.deepcopy()
        self.obs = obs

    def step(self):
        return True 
    
    def step_force(self):
        tries = 1
        while self.step()==False:
            tries += 1
            pass
        return tries

#create a static lnprob function to pass to the emcee package
def lnprob(x, e):
    e.state.set_params(x)
    try:
        logp = e.state.get_logp(e.obs)
    except:
        print "Collision! {t}".format(t=datetime.utcnow())
        return -np.inf
    return logp

'''
emcee MCMC coupled with rebound.
'''
class Ensemble(Mcmc):
    def __init__(self, initial_state, obs, scales, nwalkers=10):
        super(Ensemble,self).__init__(initial_state, obs)
        self.set_scales(scales)
        self.nwalkers = nwalkers
        self.states = [self.state.get_params() for i in range(nwalkers)]
        self.previous_states = [self.state.get_params() for i in range(nwalkers)]
        self.lnprob = None
        self.totalErrorCount = 0
        for i,s in enumerate(self.states):
            shift = 0.1e-2*self.scales*np.random.normal(size=self.state.Nvars)
            self.states[i] += shift
        self.sampler = emcee.EnsembleSampler(nwalkers,self.state.Nvars, lnprob, args=[self])

    '''
    Constitutes 1 emcee step.
    '''
    def step(self):
        self.previous_states = self.states
        self.states, self.lnprob, rstate = self.sampler.run_mcmc(self.states,1,lnprob0=self.lnprob)
        for i in range(len(self.states)):
            for j in range(len(self.states[0])):
                if(self.previous_states[i][j] != self.states[i][j]):
                    return True
        else:
            return False

    '''
    Sets the scales for the initial random distribution of walkers. Mileage may vary.
    '''
    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]

'''
Metropolis-Hastings MCMC coupled with rebound.
'''
class Mh(Mcmc):
    def __init__(self, initial_state, obs):
        super(Mh,self).__init__(initial_state, obs)
        #default value of 3e-5 for MH mcmc
        self.step_size = 3e-5

    '''
    Generates a proposal randomly.
    '''
    def generate_proposal(self):
        prop = self.state.deepcopy()
        shift = self.step_size*self.scales*np.random.normal(size=self.state.Nvars)
        prop.shift_params(shift)
        return prop

    '''
    Sets the scales used in MH's random steps.
    '''
    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]
    '''
    Constitutes 1 MH step. Generates a proposal/transitions/catch errors & collisions.
    '''
    def step(self):
        while True:
            try:
                logp = self.state.get_logp(self.obs)
                proposal = self.generate_proposal() 
                if (proposal.priorHard()):
                    return False
                logp_proposal = proposal.get_logp(self.obs)
                if np.exp(logp_proposal-logp)>np.random.uniform():
                    self.state = proposal
                    return True
                return False
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                return False

'''
smala MCMC coupled with rebound.
'''
class Smala(Mcmc):
    def __init__(self, initial_state, obs, eps, alp):
        super(Smala,self).__init__(initial_state, obs)
        self.epsilon = eps
        self.alpha = alp

    '''
    Soft absolute metric, makes sure the eigenvalues are positive.
    '''
    def softabs(self, hessians):
        lam, Q = np.linalg.eig(-hessians)
        lam_twig = lam*1./np.tanh(self.alpha*lam)
        H_twig = np.dot(Q,np.dot(np.diag(lam_twig),Q.T))    
        return H_twig

    '''
    Generates a proposal based on the last state's logp, logp_d, logp_dd values.
    '''
    def generate_proposal(self):
        logp, logp_d, logp_dd = self.state.get_logp_d_dd(self.obs) 
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        Ginvsqrt = np.linalg.cholesky(Ginv)   

        mu = self.state.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        newparams = mu + self.epsilon * np.dot(Ginvsqrt, np.random.normal(0.,1.,self.state.Nvars))
        prop = self.state.deepcopy()
        prop.set_params(newparams)
        return prop

    '''
    Calculates the transition probability given from a state to another.
    '''
    def transitionProbability(self,state_from, state_to):
        logp, logp_d, logp_dd = state_from.get_logp_d_dd(self.obs) 
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        mu = state_from.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        return stats.multivariate_normal.logpdf(state_to.get_params(),mean=mu, cov=(self.epsilon)**2*Ginv)
        
    '''
    Constitutes 1 smala step. Generates a proposal/transitions/catch errors & collisions.
    '''
    def step(self):
        while True:
            try: 
                stateStar = self.generate_proposal()
                if (stateStar.priorHard()):
                    return False
                q_ts_t = self.transitionProbability(self.state, stateStar)
                q_t_ts = self.transitionProbability(stateStar, self.state)
                break
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                return False
            except np.linalg.linalg.LinAlgError as err:
                print "np.linalg.linalg.LinAlgErrorhas occured, investigate later..."
                print stateStar.get_params()
                print self.state.get_params()
                quit()
        if np.exp(stateStar.logp-self.state.logp+q_t_ts-q_ts_t) > np.random.uniform():
            self.state = stateStar
            return True
        return False



class Alsmala(Smala):
    def __init__(self, initial_state, obs, eps, alp):
        super(Alsmala,self).__init__(initial_state, obs, eps, alp)

    def generate_proposal_mala(self):
        logp, logp_d, logp_dd = self.state.get_logp(self.obs), self.state.logp_d, self.state.logp_dd
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        Ginvsqrt = np.linalg.cholesky(Ginv)   

        mu = self.state.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        newparams = mu + self.epsilon * np.dot(Ginvsqrt, np.random.normal(0.,1.,self.state.Nvars))
        prop = self.state.deepcopy()
        prop.set_params(newparams)
        prop.logp_d = logp_d
        prop.logp_dd = logp_dd
        return prop

    def transitionProbability_mala(self,state_from, state_to):
        logp, logp_d, logp_dd = state_from.get_logp(self.obs), state_from.logp_d, state_from.logp_dd 
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        mu = state_from.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        return stats.multivariate_normal.logpdf(state_to.get_params(),mean=mu, cov=(self.epsilon)**2*Ginv)

    def step_mala(self):
        while True:
            try: 
                stateStar = self.generate_proposal_mala()
                if (stateStar.priorHard()):
                    return False
                q_ts_t = self.transitionProbability_mala(self.state, stateStar)
                q_t_ts = self.transitionProbability_mala(stateStar, self.state)
                break
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                return False
            except np.linalg.linalg.LinAlgError as err:
                print "np.linalg.linalg.LinAlgErrorhas occured, investigate later..."
                print stateStar.get_params()
                print self.state.get_params()
                quit()
        if np.exp(stateStar.logp-self.state.logp+q_t_ts-q_ts_t) > np.random.uniform():
            self.state = stateStar
            return True
        return False

