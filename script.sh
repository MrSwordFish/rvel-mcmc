#!/bin/bash
# MOAB/Torque submission script for multiple serial jobs on
# SciNet GPC
#
#PBS -l nodes=1:ppn=8,walltime=48:00:00
#PBS -N serialx8
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR
 
# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1
 
# EXECUTION COMMAND; ampersand off 8 jobs and wait
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p0.py) &
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p1.py) &
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p2.py) &
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p3.py) &
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p4.py) &
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p5.py) &
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p6.py) &
(cd /scratch/r/rein/rleblanc/rebound; python ./speedtest_2p7.py) &
wait