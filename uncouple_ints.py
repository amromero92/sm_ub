
import sys
import numpy as np
import itertools as it

## START 

#Interaction filename, number of neutrons and protons. Last two needed for scaling
interaction_file, N, Z = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def read_mels(inputfile,N,Z):   #Read the matrix elements from inputfile written in ANTOINE's format. N,Z are the valence neutrons and protons needed for scaling  

   global zepsi, idens, core, zmass, qmax, VabcdJT, abcdJT, particles 
   global number_of_shells, shells, shells_n, shells_l, shells_j, shells_2j

   f = open(inputfile,'r')

   interaction = f.readline()
   print('Reading the matrix elements of interaction: ' + interaction)

   shell_info = f.readline().strip().split()
   qmax = int(shell_info[0])
   number_of_shells = int(shell_info[1])
   shells = [int(i) for i in shell_info[2:2+number_of_shells] ]

   print('Model space')
   print('===============')
   for j in range(len(shells)):
     print(j+1, shells[j]) 
   print('===============')

   shells_n, shells_l, shells_2j = [], [], []

   for i in shells:
      shstr = str(i)
      strlen = len(shstr)
      if (strlen == 3):
         shells_n.append(0)                               #n values
         shells_l.append(int(shstr[0]))                   #l values 
         shells_2j.append(float(shstr[1:3]))              #2j values
      else: 
         shells_n.append(int(shstr[0]))                   #n values
         shells_l.append(int(shstr[1]))                   #l values
         shells_2j.append(float(shstr[2:4]))              #2j values

      shells_j = [ 0.5 * j for j in shells_2j ]           #j values


   if (qmax == 1):
     zepsi_info = f.readline().strip().split()
     zepsi = [ float(i) for i in zepsi_info ]                   #Single-particle energies
   elif (qmax == 2): 
     print("qmax !=1 not implemented yet!")
     sys.exit()
     #zepsi_info_1 = f.readline().strip().split()
     #zepsi_1 = [ float(i) for i in zepsi_info_1 ]                   #Single-particle energies (space 1)
     #zepsi_info_2 = f.readline().strip().split()
     #zepsi_2 = [ float(i) for i in zepsi_info_2 ]                   #Single-particle energies (space 2)

   core_info = f.readline().strip().split()

   idens, core, zmass = int(core_info[0]), [int(i) for i in core_info[1:3] ], [float(i) for i in core_info[3:5] ] 
   
   assert core[0] == core[1]

   particles = N + Z + core[0]
   
   if (idens == 1 or idens == 2):
    print("Mass dependence IS TAKEN into account explicitly, with neutron and proton number= " + str(N) + " " + str(Z))
   if (idens == 0):
    print("Mass dependence is not taken into account explicitly, matrix elements will be the same for all nuclei in this valence space")    

   #Start the loop over the matrix elements. They are coupled to JT, you need to uncouple them later on
 
   VabcdJT = []    #a,b,c,d are the four different shells of the two-body interaction
   abcdJT = []     #Store the information of the matrix elements, for uncoupling later on
   V_info = f.readlines()
   counter = 0

   while (counter < len(V_info)):
 
    V_line_info = V_info[counter].strip().split()
 
    tmin, tmax = int(V_line_info[0]), int(V_line_info[1])
 
    a, b, c, d = int(V_line_info[2]), int(V_line_info[3]), int(V_line_info[4]), int(V_line_info[5])
 
    jmin, jmax = int(V_line_info[6]), int(V_line_info[7])
 
    t_range = range(tmin,tmax+1)
    j_range = range(jmin,jmax+1)
 
    for i in range(len(t_range)):
      counter += 1
      V_mel_line = V_info[counter].strip().split()
      for j in range(len(j_range)):
        VabcdJT.append([float(V_mel_line[j]),a,b,c,d,j_range[j],t_range[i]])
        abcdJT.append([a,b,c,d,j_range[j],t_range[i]])
    counter += 1

   #For completeness, we include in VabcdJT the permutations of the matrix elements 
   #with corresponding phase. Only for non zero values
   VabcdJTcopy = list(VabcdJT)
   abcdJTcopy = list(abcdJT)
   for a in shells:
    for b in shells: 
     for c in shells:
      for d in shells:
       for j in range(0,int(max(shells_2j))+1): 
        for t in 0,1:

          if ([a,b,c,d,j,t] in abcdJTcopy):
             ind = abcdJT.index([a,b,c,d,j,t])
             Vmel = VabcdJTcopy[ind][0]
             ja, jb = shells_j[shells.index(a)], shells_j[shells.index(b)]
             jc, jd = shells_j[shells.index(c)], shells_j[shells.index(d)]
             phab, phcd = (-1) ** int(ja + jb + j + t), (-1) ** int(jc + jd + j + t)
             #Formula (8.31) from Suhonen
             if (abs(Vmel) > 0.00000000001):

              if ( [a,b,d,c] != [a,b,c,d] ): 
               VabcdJT.append([Vmel * phcd,a,b,d,c,j,t])
               abcdJT.append([a,b,d,c,j,t])

              if ( [b,a,c,d] != [a,b,c,d] ): 
               VabcdJT.append([Vmel * phab,b,a,c,d,j,t])
               abcdJT.append([b,a,c,d,j,t])

              if ( [b,a,d,c] != [a,b,c,d] ): 
               VabcdJT.append([Vmel * phab * phcd,b,a,d,c,j,t])
               abcdJT.append([b,a,d,c,j,t])

              if ( [c,d,a,b] != [a,b,c,d] ): 
               VabcdJT.append([Vmel,c,d,a,b,j,t])
               abcdJT.append([c,d,a,b,j,t])

              if ( [c,d,a,b] != [a,b,c,d] ): 
               VabcdJT.append([Vmel * phab,c,d,b,a,j,t])
               abcdJT.append([c,d,b,a,j,t])

              if ( [d,c,a,b] != [a,b,c,d] ): 
               VabcdJT.append([Vmel * phcd,d,c,a,b,j,t])
               abcdJT.append([d,c,a,b,j,t])

              if ( [d,c,b,a] != [a,b,c,d] ): 
               VabcdJT.append([Vmel * phab * phcd,d,c,b,a,j,t])
               abcdJT.append([d,c,b,a,j,t])

   return 

def read_j2t2(inputj2,inputt2):   #Read the matrix elements from inputfile written in ANTOINE's format 

   global VJ2abcdJT, J2abcdJT, VT2abcdJT, T2abcdJT, shells, shells_n, shells_l, shells_2j, shells_j  

   f = open(inputj2,'r')

   interaction = f.readline()
   print('Reading the matrix elements of interaction: ' + interaction)

   shell_info = f.readline().strip().split()
   number_of_shells = int(shell_info[0])
   shells = [int(i) for i in shell_info[1:1+number_of_shells] ]

   shells_n, shells_l, shells_2j = [], [], []

   for i in shells:
      shstr = str(i)
      strlen = len(shstr)
      if (strlen == 3):
         shells_n.append(0)                               #n values
         shells_l.append(int(shstr[0]))                   #l values 
         shells_2j.append(float(shstr[1:3]))              #2j values
      else: 
         shells_n.append(int(shstr[0]))                   #n values
         shells_l.append(int(shstr[1]))                   #l values
         shells_2j.append(float(shstr[2:4]))              #2j values

      shells_j = [ 0.5 * j for j in shells_2j ]           #j values


   zepsi_info = f.readline().strip().split()
   zepsi = [ float(i) for i in zepsi_info ]                   #Single-particle mels


   #Start the loop over the matrix elements. They are coupled to JT, you need to uncouple them later on
 
   VJ2abcdJT = []    #a,b,c,d are the four different shells of the two-body interaction
   J2abcdJT = []     #Store the information of the matrix elements, for uncoupling later on
   V_info = f.readlines()
   counter = 0

   while (counter < len(V_info)):
 
    V_line_info = V_info[counter].strip().split()
 
    tmin, tmax = int(V_line_info[0]), int(V_line_info[1])
 
    a, b, c, d = int(V_line_info[2]), int(V_line_info[3]), int(V_line_info[4]), int(V_line_info[5])
 
    jmin, jmax = int(V_line_info[6]), int(V_line_info[7])
 
    t_range = range(tmin,tmax+1)
    j_range = range(jmin,jmax+1)
 
    for i in range(len(t_range)):
      counter += 1
      V_mel_line = V_info[counter].strip().split()
      for j in range(len(j_range)):
        VJ2abcdJT.append([float(V_mel_line[j]),a,b,c,d,j_range[j],t_range[i]])
        J2abcdJT.append([a,b,c,d,j_range[j],t_range[i]])
    counter += 1

   #For completeness, we include in VabcdJT the permutations of the matrix elements 
   #with corresponding phase. Only for non zero values
   VJ2abcdJTcopy = list(VJ2abcdJT)
   J2abcdJTcopy = list(J2abcdJT)
   for a in shells:
    for b in shells: 
     for c in shells:
      for d in shells:
       for j in range(0,int(max(shells_2j))+1): 
        for t in 0,1:

          if ([a,b,c,d,j,t] in J2abcdJTcopy):
             ind = J2abcdJT.index([a,b,c,d,j,t])
             Vmel = VJ2abcdJTcopy[ind][0]
             ja, jb = shells_j[shells.index(a)], shells_j[shells.index(b)]
             jc, jd = shells_j[shells.index(c)], shells_j[shells.index(d)]
             phab, phcd = (-1) ** int(ja + jb + j + t), (-1) ** int(jc + jd + j + t)
             #Formula (8.31) from Suhonen
             if (abs(Vmel) > 0.00000000001):

              if ( [a,b,d,c] != [a,b,c,d] ): 
               VJ2abcdJT.append([Vmel * phcd,a,b,d,c,j,t])
               J2abcdJT.append([a,b,d,c,j,t])

              if ( [b,a,c,d] != [a,b,c,d] ): 

               VJ2abcdJT.append([Vmel * phab,b,a,c,d,j,t])
               J2abcdJT.append([b,a,c,d,j,t])

              if ( [b,a,d,c] != [a,b,c,d] ): 
               VJ2abcdJT.append([Vmel * phab * phcd,b,a,d,c,j,t])
               J2abcdJT.append([b,a,d,c,j,t])

              if ( [c,d,a,b] != [a,b,c,d] ): 
               VJ2abcdJT.append([Vmel,c,d,a,b,j,t])
               J2abcdJT.append([c,d,a,b,j,t])

              if ( [c,d,a,b] != [a,b,c,d] ): 
               VJ2abcdJT.append([Vmel * phab,c,d,b,a,j,t])
               J2abcdJT.append([c,d,b,a,j,t])

              if ( [d,c,a,b] != [a,b,c,d] ): 
               VJ2abcdJT.append([Vmel * phcd,d,c,a,b,j,t])
               J2abcdJT.append([d,c,a,b,j,t])

              if ( [d,c,b,a] != [a,b,c,d] ): 
               VJ2abcdJT.append([Vmel * phab * phcd,d,c,b,a,j,t])
               J2abcdJT.append([d,c,b,a,j,t])


   f.close()

   f = open(inputt2,'r')

   interaction = f.readline()
   print('Reading the matrix elements of interaction: ' + interaction)

   shell_info = f.readline().strip().split()


   zepsi_info = f.readline().strip().split()  
   zepsi = [ float(i) for i in zepsi_info ]                   #Single-particle mels



   #Start the loop over the matrix elements. They are coupled to JT, you need to uncouple them later on
 
   VT2abcdJT = []    #a,b,c,d are the four different shells of the two-body interaction
   T2abcdJT = []     #Store the information of the matrix elements, for uncoupling later on
   V_info = f.readlines()

   counter = 0

   while (counter < len(V_info)):
 
    V_line_info = V_info[counter].strip().split()
 
    tmin, tmax = int(V_line_info[0]), int(V_line_info[1])
 
    a, b, c, d = int(V_line_info[2]), int(V_line_info[3]), int(V_line_info[4]), int(V_line_info[5])
 
    jmin, jmax = int(V_line_info[6]), int(V_line_info[7])
 
    t_range = range(tmin,tmax+1)
    j_range = range(jmin,jmax+1)
 
    for i in range(len(t_range)):
      counter += 1
      V_mel_line = V_info[counter].strip().split()
      for j in range(len(j_range)):
        VT2abcdJT.append([float(V_mel_line[j]),a,b,c,d,j_range[j],t_range[i]])
        T2abcdJT.append([a,b,c,d,j_range[j],t_range[i]])
    counter += 1

   #For completeness, we include in VabcdJT the permutations of the matrix elements 
   #with corresponding phase. Only for non zero values
   VT2abcdJTcopy = list(VT2abcdJT)
   T2abcdJTcopy = list(T2abcdJT)
   for a in shells:
    for b in shells: 
     for c in shells:
      for d in shells:
       for j in range(0,int(max(shells_2j))+1): 
        for t in 0,1:

          if ([a,b,c,d,j,t] in T2abcdJTcopy):
             ind = T2abcdJT.index([a,b,c,d,j,t])
             Vmel = VT2abcdJTcopy[ind][0]
             ja, jb = shells_j[shells.index(a)], shells_j[shells.index(b)]
             jc, jd = shells_j[shells.index(c)], shells_j[shells.index(d)]
             phab, phcd = (-1) ** int(ja + jb + j + t), (-1) ** int(jc + jd + j + t)
             #Formula (8.31) from Suhonen
             if (abs(Vmel) > 0.00000000001):

              if ( [a,b,d,c] != [a,b,c,d] ): 
               VT2abcdJT.append([Vmel * phcd,a,b,d,c,j,t])
               T2abcdJT.append([a,b,d,c,j,t])

              if ( [b,a,c,d] != [a,b,c,d] ): 
               VT2abcdJT.append([Vmel * phab,b,a,c,d,j,t])
               T2abcdJT.append([b,a,c,d,j,t])

              if ( [b,a,d,c] != [a,b,c,d] ): 
               VT2abcdJT.append([Vmel * phab * phcd,b,a,d,c,j,t])
               T2abcdJT.append([b,a,d,c,j,t])

              if ( [c,d,a,b] != [a,b,c,d] ): 
               VT2abcdJT.append([Vmel,c,d,a,b,j,t])
               T2abcdJT.append([c,d,a,b,j,t])

              if ( [c,d,a,b] != [a,b,c,d] ): 
               VT2abcdJT.append([Vmel * phab,c,d,b,a,j,t])
               T2abcdJT.append([c,d,b,a,j,t])

              if ( [d,c,a,b] != [a,b,c,d] ): 
               VT2abcdJT.append([Vmel * phcd,d,c,a,b,j,t])
               T2abcdJT.append([d,c,a,b,j,t])

              if ( [d,c,b,a] != [a,b,c,d] ): 
               VT2abcdJT.append([Vmel * phab * phcd,d,c,b,a,j,t])
               T2abcdJT.append([d,c,b,a,j,t])          

   return 

def HO_levels(shells):    #Defines the HO basis: |shell,n,l,j,mj,mt>

    global HO_levels, dim_space, dim_space_p, dim_space_n

    HO_levels = []
    for i in range(len(shells)):
     n = shells_n[i]
     l = shells_l[i]
     j = shells_j[i]
     e = zepsi[i]
     for k in range(int(2*j+1)):
      m = j-k
      
      HO_levels.append([shells[i],n,l,j,m,-0.5,e])        #Protons with tz =-1/2

    for i in range(len(shells)):
     n = shells_n[i]
     l = shells_l[i]
     j = shells_j[i]
     e = zepsi[i]
     for k in range(int(2*j+1)):
      m = j-k
      
      HO_levels.append([shells[i],n,l,j,m,+0.5,e])        #Neutrons with tz =+1/2

    dim_space = len(HO_levels)
    dim_space_p, dim_space_n = dim_space/2, dim_space/2
    
    #Write single-particle data to file 
    ff = open('sp.dat','w')
    ff.write('  #  n l    j    m t_z      eps\n')
    c = 0
    for i in HO_levels:
     if (i[5] == -0.5):
      tz = 'p'
     if (i[5] == 0.5):
      tz = 'n'
     ff.write('%3i %2i %i %4.1f %4.1f %3s %8.5f \n' % (c, i[1],i[2],i[3],i[4],tz,i[6]))
     c += 1
    ff.close()
    print("Single-particle information stored in sp.dat")

    return HO_levels

#Clebsch-Gordan coefficients. Adapted from HFODD 10.1088/1361-6471/ac0a82
def cg(J1,M1,J2,M2,J,M): #YOU HAVE TO ENTER 2*J1, 2*M1... ALL INTEGERS 
    
  assert (isinstance(J1, int) == True)
  assert (isinstance(M1, int) == True)
  assert (isinstance(J2, int) == True)
  assert (isinstance(M2, int) == True)
  assert (isinstance(J, int) == True)
  assert (isinstance(M, int) == True)

  MV25=150
  LSIZ=MV25

  H, L = np.zeros(MV25,dtype=float), np.zeros(MV25,dtype=int)
  H[0] = 1.0
  L[0] = 0
  LAST = 1
  SCALE1 = 8.0 

  cgcoef=0.0
  
  if (J < 0):
   print('J<0')
   return cgcoef

  if (J1 < 0): 
   print('J1<0')
   return cgcoef      

  if (J2 < 0):
   print('J2<0')
   return cgcoef

  if (abs(M) > J):
   print('abs(M) > J')
   return cgcoef

  if (abs(M1) > J1):
   print('abs(M1) > J1')
   return cgcoef

  if (abs(M2) > J2):
   print('abs(M2) > J2')
   return cgcoef                   

  if (M1 + M2 != M):
   print('M1+M2 != M')
   return cgcoef

  if (np.mod(J1+J2+J,2) != 0):
   print('J1+J2+J NOT EVEN')
   return cgcoef


  I0=int((J1+J2+J)/2+1)

  if (J1+J2-J < 0):
   print('J1+J2-J < 0')
   return cgcoef

  I1=int((J1+J2-J)/2)

  if (J < abs(J1-J2)):
   print('J < |J1-J2|')
   return cgcoef   
      
  I2=int((J-(J1-J2))/2)
  I3=int((J+(J1-J2))/2)

  if (np.mod(J -M ,2) != 0):  
   print('MOD(J-M,2) != 0')
   return cgcoef                    
  
  I8=int((J +M )/2)
  I9=int((J -M )/2)


  if (np.mod(J1-M1,2) != 0):
   print('MOD(J1-M1,2) != 0')
   return cgcoef
                       
  I4=int((J1+M1)/2)
  I5=int((J1-M1)/2)
      
  if (np.mod(J2-M2,2) != 0):
   print('MOD(J2-M2,2) != 0')
   return cgcoef

  I6=int((J2+M2)/2)
  I7=int((J2-M2)/2)
  N2=J2-J-M1
  N3=J1-J+M2
  N4=min(I1,I5,I6)
  N5=max(0,N2,N3)
  N1=int(N5/2)
    

  if (N1 > N4):
   print('N1 > N4')
   return cgcoef                 
      
  MM1=max(I0,I1,I2,I3,I4,I5,I6,I7,I8,I9,N4-int(N2/2),N4-int(N3/2))+1

  if (MM1 <= LAST):

    IY =  (L[I1]+L[I2]+L[I3]+L[I4]+L[I5]+L[I6]+L[I7]+L[I8]+L[I9]-L[I0])/2

    Y=np.sqrt(H[I1]*H[I2]*H[I3]*H[I4]*H[I5]*H[I6]*H[I7]*H[I8]*H[I9]/H[I0]*(J+1))
      
        
    if (np.mod(N5,4) != 0): Y=-Y

    Z=0.0

    for N5 in range(N1,N4):
     MM1=I1-N5
     MM2=I5-N5
     MM3=I6-N5
     MM4=N5-N2/2
     MM5=N5-N3/2
      
     X=Y/H[MM1]/H[MM2]/H[MM3]/H[MM4]/H[MM5]/H[N5]
     IX= L[MM1]+L[MM2]+L[MM3]+L[MM4]+L[MM5]+L[N5]

     if (IY-IX < 0):
      X=X/SCALE1
      IX=IX-1
     if (IY-IX > 0):
      X=X*SCALE1
      IX=IX+1
     if (IY-IX == 0):
      Z=Z+X

    Y=-Y
    cgcoef=Z
    return cgcoef
    

  if (MM1 >= LSIZ):
    print('ERROR: larger factorials are needed')
    print('Increase the size MV25 of arrays H and L in routine cg')
    sys.exit()

 
  X=LAST
  MM2=LAST+1
  LAST=MM1
  for MM3 in range(MM2,MM1+1):
    H[MM3-1]=H[MM3-2]*X
    L[MM3-1]=L[MM3-2]
    if ( H[MM3-1] < SCALE1 ):
      X=X+1.0 
      continue 
    while ( H[MM3-1] > SCALE1 ):
      H[MM3-1]=H[MM3-1]/SCALE1**2
      L[MM3-1]=L[MM3-1]+2
      if ( H[MM3-1] < SCALE1 ):
       X=X+1.0 
       continue 

  IY =  int((L[I1]+L[I2]+L[I3]+L[I4]+L[I5]+L[I6]+L[I7]+L[I8]+L[I9]-L[I0])/2)

  Y=np.sqrt(H[I1]*H[I2]*H[I3]*H[I4]*H[I5]*H[I6]*H[I7]*H[I8]*H[I9]/H[I0]*(J+1))
    
  if (np.mod(N5,4) != 0): Y=-Y
  
  Z=0.0

  for N5 in range(N1,N4+1):
   MM1=I1-N5
   MM2=I5-N5
   MM3=I6-N5
   MM4=N5-int(N2/2)
   MM5=N5-int(N3/2)
      
   X=Y/H[MM1]/H[MM2]/H[MM3]/H[MM4]/H[MM5]/H[N5]
   IX= L[MM1]+L[MM2]+L[MM3]+L[MM4]+L[MM5]+L[N5]

   while (IY-IX < 0):
    X=X/SCALE1
    IX=IX-1
   while (IY-IX > 0):
    X=X*SCALE1
    IX=IX+1
   if (IY-IX == 0):
    Z=Z+X

   Y=-Y

  cgcoef=Z
    
  return cgcoef    

def krond(a,b):   #Kronecker delta
    krond = 0
    if (a == b):
      krond = 1
    return krond

#Based on (8.24) from Suhonen
def uncouple_mels(i_a,i_b,i_c,i_d):        #i_a,i_b,i_c,i_d are the indices of HO levels

   a, b, c, d = HO_levels[i_a][0], HO_levels[i_b][0], HO_levels[i_c][0], HO_levels[i_d][0]

   ja, jb, jc, jd = HO_levels[i_a][3], HO_levels[i_b][3], HO_levels[i_c][3], HO_levels[i_d][3]

   ma, mb, mc, md = HO_levels[i_a][4], HO_levels[i_b][4], HO_levels[i_c][4], HO_levels[i_d][4]

   ta, tb, tc, td = HO_levels[i_a][5], HO_levels[i_b][5], HO_levels[i_c][5], HO_levels[i_d][5]

   assert (ma + mb == mc + md)
   assert (ta + tb == tc + td)

   delta_ab, delta_cd = krond(a,b), krond(c,d)

   uncouple_mel = 0.0

   jmin = int(min(abs(ja-jb), abs(jc-jd)))
   jmax = int(max(ja+jb, jc+jd))

   tmin = 0
   tmax = 1

   for j in range(jmin, jmax+1):
     if ( abs(ja-jb) > j or abs(jc-jd) > j): continue
     if ( ja+jb < j or jc+jd < j ): continue
     if ( abs(ma+mb) > j or abs(mc+md) > j ): continue
     cgmab = cg(int(2*ja),int(2*ma),int(2*jb),int(2*mb),int(2*j),int(2*(ma+mb)))
     cgmcd = cg(int(2*jc),int(2*mc),int(2*jd),int(2*md),int(2*j),int(2*(mc+md)))


     for t in range(tmin,tmax+1):
      if ( abs(ta+tb) > t or abs(tc+td) > t ): continue
      cgtab = cg(int(2*1/2),int(2*ta),int(2*1/2),int(2*tb),2*t,int(2*(ta+tb)))
      cgtcd = cg(int(2*1/2),int(2*tc),int(2*1/2),int(2*td),2*t,int(2*(tc+td)))


      phasJT = (-1.0) ** (j + t)
      ttz = int(ta + tb + 2*t)
      nab = np.sqrt(1.0 - delta_ab * phasJT) / (1.0 + delta_ab)
      ncd = np.sqrt(1.0 - delta_cd * phasJT) / (1.0 + delta_cd)

      if ( (abs(nab) == 0.0) or (abs(ncd) == 0.0)):
         continue 

      ninv = 1.0 / (nab * ncd)

      if ([a,b,c,d,j,t] in abcdJT):
        uncouple_mel += ninv * cgmab * cgmcd * cgtab * cgtcd * VabcdJT[abcdJT.index([a,b,c,d,j,t])][0]


   return uncouple_mel

def uncouple_j2(i_a,i_b,i_c,i_d):        #i_a,i_b,i_c,i_d are the indices of HO levels

   a, b, c, d = HO_levels[i_a][0], HO_levels[i_b][0], HO_levels[i_c][0], HO_levels[i_d][0]

   na, nb, nc, nd = HO_levels[i_a][1], HO_levels[i_b][1], HO_levels[i_c][1], HO_levels[i_d][1]

   ja, jb, jc, jd = HO_levels[i_a][3], HO_levels[i_b][3], HO_levels[i_c][3], HO_levels[i_d][3]

   ma, mb, mc, md = HO_levels[i_a][4], HO_levels[i_b][4], HO_levels[i_c][4], HO_levels[i_d][4]

   ta, tb, tc, td = HO_levels[i_a][5], HO_levels[i_b][5], HO_levels[i_c][5], HO_levels[i_d][5]


   assert (ma + mb == mc + md)
   assert (ta + tb == tc + td)

   delta_ab, delta_cd = krond(a,b), krond(c,d)

   uncouple_melj2 = 0.0

   jmin = int(min(abs(ja-jb), abs(jc-jd)))
   jmax = int(max(ja+jb, jc+jd))

   tmin = 0
   tmax = 1

   for j in range(jmin, jmax+1):
     if ( abs(ja-jb) > j or abs(jc-jd) > j): continue
     if ( ja+jb < j or jc+jd < j ): continue
     if ( abs(ma+mb) > j or abs(mc+md) > j ): continue
     cgmab = cg(int(2*ja),int(2*ma),int(2*jb),int(2*mb),int(2*j),int(2*(ma+mb)))
     cgmcd = cg(int(2*jc),int(2*mc),int(2*jd),int(2*md),int(2*j),int(2*(mc+md)))


     for t in range(tmin,tmax+1):
      if ( abs(ta+tb) > t or abs(tc+td) > t ): continue
      cgtab = cg(int(2*1/2),int(2*ta),int(2*1/2),int(2*tb),2*t,int(2*(ta+tb)))
      cgtcd = cg(int(2*1/2),int(2*tc),int(2*1/2),int(2*td),2*t,int(2*(tc+td)))




      phasJT = (-1.0) ** (j + t)
      ttz = int(ta + tb + 2*t)
      nab = np.sqrt(1.0 - delta_ab * phasJT) / (1.0 + delta_ab)
      ncd = np.sqrt(1.0 - delta_cd * phasJT) / (1.0 + delta_cd)

      if ( (abs(nab) == 0.0) or (abs(ncd) == 0.0)):
         continue 

      ninv = 1.0 / (nab * ncd)

      if ([a,b,c,d,j,t] in J2abcdJT):
        uncouple_melj2 += ninv * cgmab * cgmcd * cgtab * cgtcd * VJ2abcdJT[J2abcdJT.index([a,b,c,d,j,t])][0]

   return uncouple_melj2


def uncouple_t2(i_a,i_b,i_c,i_d):        #i_a,i_b,i_c,i_d are the indices of HO levels

   a, b, c, d = HO_levels[i_a][0], HO_levels[i_b][0], HO_levels[i_c][0], HO_levels[i_d][0]

   na, nb, nc, nd = HO_levels[i_a][1], HO_levels[i_b][1], HO_levels[i_c][1], HO_levels[i_d][1]

   ja, jb, jc, jd = HO_levels[i_a][3], HO_levels[i_b][3], HO_levels[i_c][3], HO_levels[i_d][3]

   ma, mb, mc, md = HO_levels[i_a][4], HO_levels[i_b][4], HO_levels[i_c][4], HO_levels[i_d][4]

   ta, tb, tc, td = HO_levels[i_a][5], HO_levels[i_b][5], HO_levels[i_c][5], HO_levels[i_d][5]


   assert (ma + mb == mc + md)
   assert (ta + tb == tc + td)

   delta_ab, delta_cd = krond(a,b), krond(c,d)

   uncouple_melj2, uncouple_melt2 = 0.0, 0.0

   jmin = int(min(abs(ja-jb), abs(jc-jd)))
   jmax = int(max(ja+jb, jc+jd))

   tmin = 0
   tmax = 1

   for j in range(jmin, jmax+1):
     if ( abs(ja-jb) > j or abs(jc-jd) > j): continue
     if ( ja+jb < j or jc+jd < j ): continue
     if ( abs(ma+mb) > j or abs(mc+md) > j ): continue
     cgmab = cg(int(2*ja),int(2*ma),int(2*jb),int(2*mb),int(2*j),int(2*(ma+mb)))
     cgmcd = cg(int(2*jc),int(2*mc),int(2*jd),int(2*md),int(2*j),int(2*(mc+md)))


     for t in range(tmin,tmax+1):
      if ( abs(ta+tb) > t or abs(tc+td) > t ): continue
      cgtab = cg(int(2*1/2),int(2*ta),int(2*1/2),int(2*tb),2*t,int(2*(ta+tb)))
      cgtcd = cg(int(2*1/2),int(2*tc),int(2*1/2),int(2*td),2*t,int(2*(tc+td)))

      phasJT = (-1.0) ** (j + t)
      ttz = int(ta + tb + 2*t)
      nab = np.sqrt(1.0 - delta_ab * phasJT) / (1.0 + delta_ab)
      ncd = np.sqrt(1.0 - delta_cd * phasJT) / (1.0 + delta_cd)

      if ( (abs(nab) == 0.0) or (abs(ncd) == 0.0)):
         continue 

      ninv = 1.0 / (nab * ncd)

      if ([a,b,c,d,j,t] in T2abcdJT):
        uncouple_melt2 += ninv * cgmab * cgmcd * cgtab * cgtcd * VT2abcdJT[T2abcdJT.index([a,b,c,d,j,t])][0]


   return uncouple_melt2



def compute_H2b():         #Defines the two-body part of the hamiltonian matrix

   print("Computing the uncoupled TBME of the Hamiltonian...")

   scaling = 1.0 
   if (idens == 1 or idens == 2):
    assert core[0] == core[1]
    scaling = ((core[0]+2)/float(particles))**zmass[0]

   H2b = []
   numtbme = 0
   for a in range(len(HO_levels)):
    for b in range(len(HO_levels)):
     for c in range(len(HO_levels)):
      for d in range(len(HO_levels)):

        la, lb, lc, ld = HO_levels[a][2], HO_levels[b][2], HO_levels[c][2], HO_levels[d][2]
        ma, mb, mc, md = HO_levels[a][4], HO_levels[b][4], HO_levels[c][4], HO_levels[d][4]
        ta, tb, tc, td = HO_levels[a][5], HO_levels[b][5], HO_levels[c][5], HO_levels[d][5]

        if ((-1) ** (la+lb) != (-1) ** (lc+ld) ):
           continue 
        if ( ma + mb != mc + md ):
           continue 
        if ( ta + tb != tc + td ):
           continue 

        Vu = uncouple_mels(a,b,c,d)
        if (abs(Vu) > 0.00000000001):
         H2b.append([scaling * Vu,a,b,c,d])
         numtbme += 1

   #Write matrix elements to file 
   ff = open('H2b.dat','w')
   for i in H2b:
     ff.write('%18.16f %i %i %i %i \n' % (i[0], i[1], i[2], i[3], i[4]))
   ff.close()
   print("Matrix elements of Hamiltonian stored in H2b.dat")   
     

   return H2b


def compute_j2t2():         #Defines the two-body part of the J^2 and T^2

   print("Computing the uncoupled TBME of J^2 and T^2...")

   J2, T2 = [], []
   numtbme = 0
   for a in range(len(HO_levels)):
    for b in range(len(HO_levels)):
     for c in range(len(HO_levels)):
      for d in range(len(HO_levels)):

        la, lb, lc, ld = HO_levels[a][2], HO_levels[b][2], HO_levels[c][2], HO_levels[d][2]
        ma, mb, mc, md = HO_levels[a][4], HO_levels[b][4], HO_levels[c][4], HO_levels[d][4]
        ta, tb, tc, td = HO_levels[a][5], HO_levels[b][5], HO_levels[c][5], HO_levels[d][5]

        if ((-1) ** (la+lb) != (-1) ** (lc+ld) ):
           continue 
        if ( ma + mb != mc + md ):
           continue 
        if ( ta + tb != tc + td ):
           continue 

        VJ = uncouple_j2(a,b,c,d)
        if (abs(VJ) > 0.00000000001):
         J2.append([VJ,a,b,c,d])
        VT = uncouple_t2(a,b,c,d)
        if (abs(VT) > 0.00000000001):
         T2.append([VT,a,b,c,d])
  

   #Write matrix elements to file 
   ff = open('J2.dat','w')
   for i in J2:
     ff.write('%18.16f %i %i %i %i \n' % (i[0], i[1], i[2], i[3], i[4]))
   ff.close()
   print("Matrix elements of J^2 stored in J2.dat")

   ff = open('T2.dat','w')
   for i in T2:
     ff.write('%18.16f %i %i %i %i \n' % (i[0], i[1], i[2], i[3], i[4]))
   ff.close()
   print("Matrix elements of T^2 stored in T2.dat")   
     
   return



#We start here

#Read matrix elements from file
read_mels(interaction_file,N,Z)
#Set space of HO levels
ho = HO_levels(shells)
compute_H2b()

## For T^2 and J^2
read_j2t2('j2.int','t2.int')
#Set space of HO levels ( if not before )
if not ho: ho = HO_levels(shells)
compute_j2t2()


