# rvel-mcmc

##Description

This is an implementation of various Markov Chain Monte Carlo methods to solve for radial velocity data. Current implentations are Metropolis-Hastings, Affine Sampler, and SMALA. The N-Body integration is handled by Rebound. Code is split up into a few .py files. Simple profiling can be done with 

'$python -m cProfile [-o output_file] [-s sort_order] mcmc_benchmark_*.py'

##Installation

To run these many packages are needed. Rebound can be installed using '$pip install rebound'. Similarly for Numpy, Matplotlib and Corner.

## License

Figure this out later.
