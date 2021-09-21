# Annealing-Samplers
A simple repository for various annealing models in the transverse-field Ising model.

## Setup
The code in this repo was tested and run in Atom's Juno IDE. The quick install file outlines the packages that are needed to run the files.

## Samplers
All current tested samplers have examples shown in ```example.jl```, with a testing tool to profile and benchmark the samplers in ```testing.jl```.

Current samplers include
1. Simulated annealing
2. Spin Vector Monte Carlo
3. Spin Vector Monte Carlo with transverse-field updates
4. Path integral Monte Carlo

The samplers are made to be as general as possible such that you can 
1.  input custom schedules and initial states for protocols like reverse annealing 
2.  use independent qubit schedules for protocols such as LSTF-DQA (or like D-Wave's h-gain tool)
3.  choose to include D-Wave's model of qubit cross-talk
4.  choose to include noisy parameter values to simulate integrated control errors

An example D-Wave schedule has also been included for general interest to the user. More samplers are to be added to this repository in the near future. 
