This repository belongs to the Master Thesis of Arend-Jan Quist (2022).
It consists of simulations on the collimation sieve and post-processing algorithm from Peikert (2020).

# Usage of the code

The python code in "C_sieve_simulator.py" is the simulator of the c-sieve of Peikert's algorithm.
Use "Plot_multipliers_c-sieve.ipynb" to reproduce the pictures of the c-sieve output as presented in the thesis.
Use "Simulator_post-processing.ipynb" to simulate the post-processing on the output of the c-sieve. To import phase vectors from the c-sieve in the post-processing simulator, use "Save_phasevectors_c-sieve_simulator.ipynb" to save the phase vectors from the c-sieve first, then the post-processing simulator can use them.


# Reference
Peikert, C. (2020). He gives C-sieves on the CSIDH. Advances in Cryptology–EUROCRYPT 2020, 12106, 463.
