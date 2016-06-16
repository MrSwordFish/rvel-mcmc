import numpy as np
import rebound
import copy

class State(object):

    def __init__(self, planets, ignore_vars=[]):
        self.planets = planets
        self.logp = None
        self.planets_vars = []
        self.Nvars = 0
        self.ignore_vars = ignore_vars
        for planet in planets:
            planet_vars = [x for x in planet.keys() if x not in ignore_vars]
            self.planets_vars.append(planet_vars)
            self.Nvars += len(planet_vars)


    def setup_sim(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        for planet in self.planets:
            sim.add(primary=sim.particles[0],**planet)
        sim.move_to_com()
        return sim

    def get_rv(self, times):
        sim = self.setup_sim()
        
        rv = np.zeros(len(times))
        for i, t in enumerate(times):
            sim.integrate(t)
            rv[i] = sim.particles[0].vx

        return rv

    def get_rv_plotting(self, obs, Npoints=150):
        times = np.linspace(obs.tb[0],obs.tf[len(obs.tf)-1],Npoints)
        return times, self.get_rv(times)

    def get_chi2(self, obs):
        rvf = self.get_rv(obs.tf)
        rvb = self.get_rv(obs.tb)        
        chi2f = 0.
        chi2b = 0.
        for i, tf in enumerate(obs.tf):
            chi2f += ((rvf[i]-obs.rvf[i])**2)/(obs.errorf[i]**2)
        for i, tb in enumerate(obs.tb):
            chi2b += ((rvb[i]-obs.rvb[i])**2)/(obs.errorb[i]**2)
        return (chi2b+chi2f)/(obs.Npoints)

    def get_logp(self, obs):
        if self.logp is None:
            self.logp = -self.get_chi2(obs)
        if (self.priorHard()):
            lnpri = -np.inf
        else:
            lnpri = 0.0
        return self.logp + lnpri
    
    def lnprior(theta):
        m, a, h, k, l = theta
        if 1e-7 < m < 0.1 and 1e-2 < a < 500.0 and h**2 + k**2 < 1.0 and -2*np.pi < l < 2*np.pi:
            return 0.0
        return -np.inf

    def shift_params(self, vec):
        self.logp = None
        if len(vec)!=self.Nvars:
            raise AttributeError("vector has wrong length")
        varindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if k not in self.ignore_vars:
                    self.planets[i][k] += vec[varindex]
                    varindex += 1
   
    def get_params(self):
        params = np.zeros(self.Nvars)
        parindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if k not in self.ignore_vars:
                    params[parindex] = self.planets[i][k]
                    parindex += 1
        return params

    def set_params(self, vec):
        self.logp = None
        if len(vec)!=self.Nvars:
            raise AttributeError("vector has wrong length")
        varindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if k not in self.ignore_vars:
                    self.planets[i][k] = vec[varindex]
                    varindex += 1
    
    def get_keys(self):
        keys = [""]*self.Nvars
        parindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if k not in self.ignore_vars:
                    keys[parindex] = "$%s_%d$"%(k,i)
                    parindex += 1
        return keys

    def get_rawkeys(self):
        keys = [""]*self.Nvars
        parindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if k not in self.ignore_vars:
                    keys[parindex] = k
                    parindex += 1
        return keys

    def deepcopy(self):
        return State(copy.deepcopy(self.planets), copy.deepcopy(self.ignore_vars))

    def var_pindex_vname(self, vindex):
        vi = 0.
        for pindex, p in enumerate(self.planets_vars):
            for v in p:
                if vindex == vi:
                    return pindex+1, v
                vi += 1


    def setup_sim_vars(self):
        sim = self.setup_sim()
        variations1 = []
        variations2 = []
        for vindex in range(self.Nvars):
            pindex, vname = self.var_pindex_vname(vindex)
            v = sim.add_variation(order=1)
            v.vary_pal(pindex,vname)
            variations1.append(v)
        
        for vindex1 in range(self.Nvars):
            for vindex2 in range(self.Nvars):
                if vindex1 >= vindex2:
                    pindex1, vname1 = self.var_pindex_vname(vindex1)
                    pindex2, vname2 = self.var_pindex_vname(vindex2)
                    v = sim.add_variation(order=2, first_order=variations1[vindex1], first_order_2=variations1[vindex2])
                    if pindex1 == pindex2:
                        v.vary_pal(pindex1,vname1,vname2)
                    variations2.append(v)

        sim.move_to_com()
        return sim, variations1, variations2

    def get_chi2_d_dd(self, obs):
        sim, variations1, variations2 = self.setup_sim_vars()
        chi2b = 0.
        chi2_db = np.zeros(self.Nvars)
        chi2_ddb = np.zeros((self.Nvars,self.Nvars))
        chi2f = 0.
        chi2_df = np.zeros(self.Nvars)
        chi2_ddf = np.zeros((self.Nvars,self.Nvars))
        fac = obs.Npoints
        for i, tf in enumerate(obs.tf):
            sim.integrate(tf)
            chi2f += (sim.particles[0].vx-obs.rvf[i])**2*1./(obs.errorf[i]**2 *fac)
            v2index = 0
            for vindex1 in range(self.Nvars):
                chi2_df[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-obs.rvf[i])*1./(obs.errorf[i]**2 *fac)
            
                for vindex2 in range(self.Nvars):
                    if vindex1 >= vindex2:
                        chi2_ddf[vindex1][vindex2] +=  2. * variations2[v2index].particles[0].vx * (sim.particles[0].vx-obs.rvf[i])*1./(obs.errorf[i]**2 * fac) + 2. * variations1[vindex1].particles[0].vx * variations1[vindex2].particles[0].vx*1./(obs.errorf[i]**2 * fac)
                        v2index += 1
                        chi2_ddf[vindex2][vindex1] = chi2_ddf[vindex1][vindex2]

        for i, tb in enumerate(obs.tb):
            sim.integrate(tb)
            chi2b += (sim.particles[0].vx-obs.rvb[i])**2*1./(obs.errorb[0]**2 * fac)
            v2index = 0
            for vindex1 in range(self.Nvars):
                chi2_db[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-obs.rvb[i])*1./(obs.errorb[i]**2 * fac)
            
                for vindex2 in range(self.Nvars):
                    if vindex1 >= vindex2:
                        chi2_ddb[vindex1][vindex2] +=  2. * variations2[v2index].particles[0].vx * (sim.particles[0].vx-obs.rvb[i])*1./(obs.errorb[i]**2 * fac) + 2. * variations1[vindex1].particles[0].vx * variations1[vindex2].particles[0].vx*1./(obs.errorb[i]**2 * fac)
                        v2index += 1
                        chi2_ddb[vindex2][vindex1] = chi2_ddb[vindex1][vindex2]

        return chi2b+chi2f, chi2_db+chi2_df, chi2_ddb+chi2_ddf

    def get_logp_d_dd(self, obs):
        if self.logp is None:
            chi, chi_d, chi_dd = self.get_chi2_d_dd(obs)
            self.logp, self.logp_d, self.logp_dd = -chi, -chi_d, -chi_dd
        return self.logp, self.logp_d, self.logp_dd

    def priorHard(self):
        for i, planet in enumerate(self.planets):
            if (self.planets[i]["a"] <=0.02) :
                print "Invalid state was proposed (a)"
                return True
            if (self.planets[i]["m"] <=5e-6) :
                print "Invalid state was proposed (m)"
                return True
            if "h" in (self.planets[i].keys()) or ("k" in self.planets[i].keys()):
                if (self.planets[i]["h"]**2 + self.planets[i]["k"]**2 >=1.0) :
                    print "Invalid state was proposed (h & k)"
                    return True
            if "ix" in (self.planets[i].keys()) or ("iy" in self.planets[i].keys()):
                if (self.planets[i]["ix"]**2 + self.planets[i]["iy"]**2 >=4.0) :
                    print "Invalid state was proposed (ix & iy)"
                    return True
            else:
                return False