## Brute force shell model diagonalization code of a hamiltonian matrix
import sys
import numpy as np
import numpy.linalg as LA
import itertools as it

#Configuration space dimension. We read from single-particle datafile
f = open('sp.dat','r')
sp_len = len( f.readlines()[1:] )
f.close()

pro_num   = 0     #Number of protons
neu_num   = 0     #Number of neutrons
Mmb       = 0.    #Total M of the Slater determinants forming the basis
numstates = 1     #Number of states to be displayed
verbose   = 0

#Read from cmd
neu_num, pro_num, Mmb, numstates, verbose = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) 

#START

print(" ")
print("BRUTE FORCE SHELL-MODEL DIAGONALIZATION IN THE M-SCHEME")
print(" ")

#Print stupid ASCII art
print(" ")
print("""         ,MMM8&&&.
    _...MMMMM88&&&&..._
 .::'''MMMMM88&&&&&&'''::.
::     MMMMM88&&&&&&     ::
'::....MMMMM88&&&&&&....::'
   `''''MMMMM88&&&&''''`
         'MMM8&&&'""")
print(" ")

#Print input
print(" ")
print("INPUT")
print(" ")
print("Single-particle data and TBME will be read from sp.dat and H2b.dat")  
print("Number of valence neutrons: " + str(neu_num))
print("Number of valence protons: " + str(pro_num))      
print("2*J_z of the Slater determinants: " + str(2*Mmb))   
print(" ")                                                       

#Open single-particle states quantum numbers, including their energies. Stored in H1b
sp_len = round(.5*sp_len) #We divide by two because we handle it in this manner
f = open('sp.dat','r')
len_H1b = len( f.readlines()[1:] )
f.close()
ff = open('sp.dat')
ff.readline() #Skip header 
H1b = []
for i in range(len_H1b):
 mel = ff.readline().strip().split() 
 H1b.append([int(mel[0]), int(mel[1]), int(mel[2]), float(mel[3]), float(mel[4]), 
              mel[5], float(mel[6])])
ff.close()

#For printing the SP states
if (verbose == 1):
 print(" ")
 print('SP states')
 for i in H1b:
  print(i)
 print(" ")
 
assert 2*sp_len == len(H1b)

#Open two-body matrix elements from file. Stored in H2b along with the sp labels used
f = open('H2b.dat','r')
len_H2b = len( f.readlines() )
f.close()
ff = open('H2b.dat')
H2b = []
indices_2b = []
for i in range(len_H2b):
 mel = ff.readline().strip().split()
 H2b.append([float(mel[0]), int(mel[1]), int(mel[2]), int(mel[3]), int(mel[4])])
 indices_2b.append([int(mel[1]), int(mel[2]), int(mel[3]), int(mel[4])])
ff.close()




#Open J^2 and T^2 matrix elements from file. Stored in J2b and T2b along with the sp labels used
f = open('J2.dat','r')
len_J2b = len( f.readlines() )
f.close()
ff = open('J2.dat')
J2b = []
indices_J2b = []
for i in range(len_J2b):
 mel = ff.readline().strip().split()
 J2b.append([float(mel[0]), int(mel[1]), int(mel[2]), int(mel[3]), int(mel[4])])
 indices_J2b.append([int(mel[1]), int(mel[2]), int(mel[3]), int(mel[4])])
ff.close()

f = open('T2.dat','r')
len_T2b = len( f.readlines() )
f.close()
ff = open('T2.dat')
T2b = []
indices_T2b = []
for i in range(len_T2b):
 mel = ff.readline().strip().split()
 T2b.append([float(mel[0]), int(mel[1]), int(mel[2]), int(mel[3]), int(mel[4])])
 indices_T2b.append([int(mel[1]), int(mel[2]), int(mel[3]), int(mel[4])])
ff.close()

## Functions to compute the two body matrix elements of the hamiltonian
def inv_intersec(ind,state):
    r = [i for i in list(state) if i not in list(ind)]
    return tuple(r)

def count_swaps_sort(state):
  css = 0
  for i in range(len(state)-1):
   for j in range(0, len(state)-i-1):
     if (state[j] > state[j+1]):
       state[j], state[j+1] = state[j+1], state[j]
       css += 1
  return css

## Compute the antisym two-body matrix elements v_{abcd}
def vmel(l,r): #Used for the two-body part of the hamiltonian
                 #l,r are many body states

 v =0.

 for c_r in range(0,len(r)):
  for d_r in range(c_r+1,len(r)):
   c, d = r[c_r], r[d_r]

   aastate = inv_intersec((c,d), r )   #If c and d are not in r, then the result is zero. Continue
   ortho_check = all(item in l for item in aastate)
   if ortho_check == False: continue 

   for a_l in range(0,len(l)):
    for b_l in range(a_l+1,len(l)):

     a, b = l[a_l], l[b_l]

     if set(list(aastate) + [a,b]) != set(l): continue  #If adding a and b to r is not equal to l, continue

     if ([a,b,c,d] in indices_2b):

      l_temp = list(r)
      l_temp[r.index(c)], l_temp[r.index(d)] = a, b

      phase_swap = count_swaps_sort(l_temp)   #Count how many swaps it takes to order the state

      index_mel = indices_2b.index([a,b,c,d])
      v += (-1)**phase_swap * H2b[index_mel][0]

 return v



## Compute the antisym two-body matrix elements v_{abcd}
def vmelj2(l,r): #Used for the two-body part of the J^2
                 #l,r are many body states

 v =0.

 for c_r in range(0,len(r)):
  for d_r in range(c_r+1,len(r)):
   c, d = r[c_r], r[d_r]

   aastate = inv_intersec((c,d), r )   #If c and d are not in r, then the result is zero. Continue
   ortho_check = all(item in l for item in aastate)
   if ortho_check == False: continue 

   for a_l in range(0,len(l)):
    for b_l in range(a_l+1,len(l)):

     a, b = l[a_l], l[b_l]

     if set(list(aastate) + [a,b]) != set(l): continue  #If adding a and b to r is not equal to l, continue

     if ([a,b,c,d] in indices_J2b):

      l_temp = list(r)
      l_temp[r.index(c)], l_temp[r.index(d)] = a, b

      phase_swap = count_swaps_sort(l_temp)   #Count how many swaps it takes to order the state

      index_mel = indices_J2b.index([a,b,c,d])
      v += (-1)**phase_swap * J2b[index_mel][0]

 return v

## Compute the antisym two-body matrix elements v_{abcd}
def vmelt2(l,r): #Used for the two-body part of the T^2
                 #l,r are many body states

 v =0.

 for c_r in range(0,len(r)):
  for d_r in range(c_r+1,len(r)):
   c, d = r[c_r], r[d_r]

   aastate = inv_intersec((c,d), r )   #If c and d are not in r, then the result is zero. Continue
   ortho_check = all(item in l for item in aastate)
   if ortho_check == False: continue 

   for a_l in range(0,len(l)):
    for b_l in range(a_l+1,len(l)):

     a, b = l[a_l], l[b_l]

     if set(list(aastate) + [a,b]) != set(l): continue  #If adding a and b to r is not equal to l, continue

     if ([a,b,c,d] in indices_T2b):

      l_temp = list(r)
      l_temp[r.index(c)], l_temp[r.index(d)] = a, b

      phase_swap = count_swaps_sort(l_temp)   #Count how many swaps it takes to order the state

      index_mel = indices_T2b.index([a,b,c,d])
      v += (-1)**phase_swap * T2b[index_mel][0]

 return v


def spe(state):   #Returns the sum of single particle energies of many body state a
  ener = 0.
  for i in range(len(state)):
    ener += H1b[state[i]][-1]

  return ener

def spj2(state):   #Returns the sum of single particle J^2 of many-body state 
  j2 = 0.
  for i in range(len(state)):
    j2i = H1b[state[i]][3] 
    j2 += j2i * (j2i + 1.)

  return j2

def spt2(state):   #Returns the sum of single particle J^2 of many-body state 
  t2 = 0.
  for i in range(len(state)):
   t2 += 0.75

  return t2

#Create all many-body states with M (M-scheme for now) 
def compute_M(state):   
  #Computes total M of a given many-body state. It is assumed that state is written as 
  #eg ('0','1','2','3') where each number represents the sp state occupied.

 M = 0.
 for k in range(len(state)):
  M += H1b[state[k]][4]

 return M 
 
#Particle number operator acting on many-body state
def Ni(i, state):
 occ = 0
 for k in range(len(state)):
  if (state[k] == i):
   occ = 1
 return occ
 
#Returns occupation probabilities of sp states in a given state
def occGS(SS):

 occupationSP = []
 for i in range(2*sp_len):
  occsp = 0.
  for k in range(len(SS)):
   occn = SS[k]
   mbstate = all_states_M[k]
   if (Ni(i, mbstate) == 1): 
    occsp = occsp + occn**2
  occupationSP.append([i,occsp])

 return occupationSP  


pro_states_labels = list(it.combinations(range(sp_len),pro_num))
neu_states_labels = list(it.combinations(range(sp_len,2*sp_len),neu_num))

num_sts_pro = len(pro_states_labels)
num_sts_neu = len(neu_states_labels)



all_states_M = []
for i in pro_states_labels:
 for j in neu_states_labels:
   if ( compute_M(i+j) == Mmb):
    all_states_M.append(i+j) 

num_sts_M = len(all_states_M)
print(" ")
print("Number of Slater determinants in the M-basis: " + str(num_sts_M))
print(" ")
if (numstates > num_sts_M): numstates = num_sts_M

#For printing the MB states
if (verbose == 1):
 ff = open('many_body_states.dat','w')
 ff.write(str(num_sts_M))
 ff.write('\n')  
 print(" ")
 print('Slater determinants')
 for i in range(num_sts_M):
  print(i+1, all_states_M[i]) 
  ff.write(str(i) + ", " + str(all_states_M[i]))
  ff.write('\n') 
 print(" ")
 print('Many-body states written to file many_body_states.dat')
 ff.close()
 

 

#Build the many-body hamiltonian
hamil = np.zeros([num_sts_M,num_sts_M])

#One body part 
for i in range(num_sts_M):
  hamil[i,i] += spe(all_states_M[i])

#Two body part
for i in range(num_sts_M):
 for j in range(i,num_sts_M):
  hamil[i,j] += vmel(all_states_M[i],all_states_M[j])
  hamil[j,i]  = hamil[i,j]

assert np.allclose(hamil, hamil.conj().transpose()) == True

#Write hamiltonian to file. Beware on how you store it
print(" ")
print("Writing hamiltonian to file hamiltonian_matrix.dat")
print(" ")
ff = open('hamiltonian_matrix.dat','w')
for i in range(hamil.shape[0]):
 for j in range(hamil.shape[1]):
  if ( abs( hamil[i,j] ) > 0.0000000000001 ):
   ff.write('%i %i %18.16f\n' % (i, j, hamil[i,j]))
ff.close()

#Diagonalize it
w,v = LA.eig(hamil)
idx = w.argsort()
w = w[idx]
v = v[:,idx]
groundstate = v[:, np.argmin(w)]
firstexcite = v[:, 1]
seconexcite = v[:, 2]




#Write groundstate to file
if (verbose == 1):
 print(" ")
 print("Writing ground state to file ground_state.dat")
 print(" ")
 ff = open('ground_state.dat','w')
 for i in range(hamil.shape[0]):
  if ( abs( groundstate[i] ) > 0.0000000000001 ):
   ff.write('%i %18.16f\n' % (i, groundstate[i]))
 ff.close()
 
 print(" ")
 print("Writing first-excited state to file first_excited.dat")
 print(" ")
 ff = open('first_excited.dat','w')
 for i in range(hamil.shape[0]):
  if ( abs( firstexcite[i] ) > 0.0000000000001 ):
   ff.write('%i %18.16f\n' % (i, firstexcite[i]))
 ff.close()
 
 print(" ")
 print("Writing second-excited state to file second_excited.dat")
 print(" ")
 ff = open('second_excited.dat','w')
 for i in range(hamil.shape[0]):
  if ( abs( seconexcite[i] ) > 0.0000000000001 ):
   ff.write('%i %18.16f\n' % (i, seconexcite[i]))
 ff.close() 
 

#Build the J^2 and T^2 matrices 
J2mat = np.zeros([num_sts_M,num_sts_M])
T2mat = np.zeros([num_sts_M,num_sts_M])

#One body part 
for i in range(num_sts_M):
  J2mat[i,i] += spj2(all_states_M[i])
  T2mat[i,i] += spt2(all_states_M[i])

#Two body part
for i in range(num_sts_M):
 for j in range(i,num_sts_M):
  J2mat[i,j] += vmelj2(all_states_M[i],all_states_M[j])
  J2mat[j,i]  = J2mat[i,j]

  T2mat[i,j] += vmelt2(all_states_M[i],all_states_M[j])
  T2mat[j,i]  = T2mat[i,j]



print(' ')
print('STATES')
print('------------')
print('#  energy      J    T')
for i in range(numstates):
 energy, state = w[i], v[:,i]
 tstsq = state.transpose().dot(T2mat.dot(state))
 jstsq = state.transpose().dot(J2mat.dot(state))
 tst = 0.5 * ( np.sqrt(4.0 * tstsq + 1) - 1 )
 jst = 0.5 * ( np.sqrt(4.0 * jstsq + 1) - 1 )

 print('%i %11.8f %4.2f %4.2f' % (i+1, energy, jst, tst))
print(' ')

#Occupation of single-particle states
if (verbose == 1):
 print(' ')
 print('Occupation probabilities of single-particle states for the ground-state (normalized to valence particles)')
 print('  # occ')
 occsnuc = occGS(groundstate)
 for i in range(len(occsnuc)):
  print('%3i %8.6f' % (occsnuc[i][0], occsnuc[i][1])) 



