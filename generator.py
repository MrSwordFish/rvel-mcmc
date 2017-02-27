from itertools import chain, combinations, product

def powerset(iterable):
    s = list(iterable)
    return map(list, chain.from_iterable(combinations(s, r) for r in range(len(s)+1, 0, -1)))

pwsets = []
pwinit = [['a', 'h', 'k', 'l', 'm'],['a', 'l', 'm']]
for i in range(len(pwinit)):
	pwsets.append(powerset(pwinit[i]))
pwsetlist = [list(elem) for elem in list(product(pwsets[0], pwsets[1]))]



for i in range(1,3):
	name = "combinations_8dim_{i}.py".format(i=i)
	code = """
import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import driver
import numpy as np
import hashlib
from datetime import datetime

np.random.seed(2017+%s)
niterEmcee = 1024*5
niterSmala = 100
initial_state = state.State(planets=[{'a': 0.2275, 'h': -0.005, 'k': 0.03, 'm': 0.00094, 'l': -1.4}, {'a': 0.3663, 'm': 0.001965, 'l': 2.15}],ignore_params=%s)
initial_state.hillRadiusFactor = 2.

#obs = driver.ReadObs('TEST_2-1_COMPACT.vels')
obs = driver.CreateObs(initial_state, 90, 1.5e-4, 4e-5, 16)

emcee, chain, chainlogp, clocktimes, h1 = driver.createEns('speed_2-1_%s_emcee',niterEmcee, initial_state, obs, 32, {"m":1.5e-3, "a":0.3, "h":0.1, "k":0.1, "l":np.pi/2.})
driver.saveAuxEmcee(h1, initial_state, 'speed_2-1_%s_emcee', niterEmcee, 32, {"m":1.5e-3, "a":0.3, "h":0.1, "k":0.1, "l":np.pi/2.})
driver.saveData(chain, 'speed_2-1_%s_emcee_chain', h1)
driver.saveData(chainlogp, 'speed_2-1_%s_emcee_chainlogp', h1)
driver.saveData(clocktimes, 'speed_2-1_%s_emcee_clocktimes', h1)

smala, chain2, chainlogp2, clocktimes2, h2 = driver.createSMALA('speed_2-1_%s_smala',niterSmala, initial_state, obs, 0.12, 1.4)
driver.saveAuxSmala(h2, initial_state, 'speed_2-1_%s_smala', niterSmala, 0.12, 1.4)
driver.saveData(chain2, 'speed_2-1_%s_smala_chain', h2)
driver.saveData(chainlogp2, 'speed_2-1_%s_smala_chainlogp', h2)
driver.saveData(clocktimes2, 'speed_2-1_%s_smala_clocktimes', h2)

#MH, chain3, chainlogp3, clocktimes3, h3 = driver.createMH('speed_2-1_MH',niterMH, initial_state, obs, {"m":1.5e-3, "a":0.3, "h":0.1, "k":0.1, "l":np.pi/2.}, 0.002)
#driver.saveAuxMH(h3, initial_state, 'speed_2-1_MH', niterMH, {"m":1.5e-3, "a":0.3, "h":0.1, "k":0.1, "l":np.pi/2.}, 0.002)
#driver.saveData(chain3, 'speed_2-1_MH_chain', h3)
#driver.saveData(chainlogp3, 'speed_2-1_MH_chainlogp', h3)
#driver.saveData(clocktimes3, 'speed_2-1_MH_clocktimes', h3)
"""
	
	with open(name, 'w') as python_file:
		python_file.write(code%(i,pwsetlist[i],i,i,i,i,i,i,i,i,i,i))