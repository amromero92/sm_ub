#!/bin/sh


interaction_file="cki"  
number_neutrons=2
number_protons=2
M_slater_basis=0
numstates=5
verbose=1

#Activate conda environment
#conda activate quantum

#Create the coupled J² and T² matrix elements in ANTOINE format
python t2j2ints.py $interaction_file 

#Uncouple the interaction, J² and T² 
python uncouple_ints.py $interaction_file $number_neutrons $number_protons

#Finally, build the hamiltonian matrix and diagonalize it to find the eigenstates
python sm_diag.py $number_neutrons $number_protons $M_slater_basis $numstates $verbose
