import numpy as np
import rebound
import state
import itertools

class Observation:
    tf = None
    tb = None
    rvf = None    
    rvb = None
    Npoints = 0
    errorf = None
    errorb = None
    t = None
    rv = None
    err = None

class FakeObservation(Observation):
    def __init__(self, state, Npoints=30, error=0., errorVar=0., tmax=1.5):
        """
        Generates fake observations. 
        """
        self.Npoints = Npoints
        self.error = error
        self.errorVar = errorVar
        sim = rebound.Simulation()
        sim.add(m=1.)
        for planet in state.planets:
            sim.add(primary=sim.particles[0],**planet)
        sim.move_to_com()
        
        self.tf = np.sort(np.random.uniform(0.,tmax/2.,self.Npoints/2.))
        self.tb = np.sort(np.random.uniform(0.,-tmax/2.,self.Npoints/2.))
        self.rvf = np.zeros(self.Npoints/2.)
        self.rvb = np.zeros(self.Npoints/2.)
        self.errorf = np.zeros(self.Npoints/2.)
        self.errorb = np.zeros(self.Npoints/2.)
        for i, tf in enumerate(self.tf):
            sim.integrate(tf)
            self.errorf[i] = error + np.random.normal(0.,self.errorVar)
            self.rvf[i] = sim.particles[0].vx + np.random.normal(0., self.errorf[i])

        for i, tb in enumerate(self.tb):
            sim.integrate(tb)
            self.errorb[i] = error + np.random.normal(0.,self.errorVar)
            self.rvb[i] = sim.particles[0].vx + np.random.normal(0., self.errorb[i])

        self.t = np.concatenate((self.tb, self.tf), axis=0)
        self.rv = np.concatenate((self.rvb, self.rvf), axis=0)
        self.err = np.concatenate((self.errorb, self.errorf), axis=0)

class Observation_FromFile(Observation):
    def __init__(self, filename='yourfile.txt', Npoints=30):
        """
        Load observations from a .vels file. 
        """
        readtimes = np.genfromtxt(filename,usecols=(0),delimiter=' ',dtype=None)
        readrvs = np.genfromtxt(filename,usecols=(1),delimiter=' ',dtype=None)
        readerrors = np.genfromtxt(filename,usecols=(2),delimiter=' ',dtype=None)
        readb, readf = np.array_split(readtimes*0.01720,2)
        shift = readb[len(readb)-1]
        self.Npoints = Npoints
        self.tf = readf - shift
        self.tb = readb - shift
        self.rvb, self.rvf = np.array_split(readrvs*3.355e-5,2)
        self.errorb, self.errorf = np.array_split(readerrors*3.355e-5,2)
        self.t = np.concatenate((self.tb, self.tf), axis=0)
        self.rv = np.concatenate((self.rvb, self.rvf), axis=0)
        self.err = np.concatenate((self.errorb, self.errorf), axis=0)