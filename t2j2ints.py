#This script generates matrix elements of J^2 and T^2 in ANTOINE's format


import itertools as it
import sys 


interaction_file = sys.argv[1]

intdata = open(interaction_file,'r')      #INTERACTION TO READ CONFIGURATION SPACE FROM

header = intdata.readline()           #Read header
shells_info=intdata.readline().strip().split()    
num_shells = int(shells_info[1])      #Number of shells in space 


shells = []
for i in range(num_shells):
 shells.append(shells_info[2+i] )

print("interaction: " + header)


j_sh = [float(int(i[-1])/2.0) for i in shells ]
listtwoshells = list(it.combinations_with_replacement(shells,2))


tempj2  = open('j2.int','w')
tempt2  = open('t2.int','w')
tempj2.write('   J^2 ' + header )
tempt2.write('   T^2 ' + header )
tempj2.write(' %i ' % num_shells)
tempt2.write(' %i ' % num_shells)
for s in range(num_shells):
 tempj2.write('%i ' % int(shells[s]))
 tempt2.write('%i ' % int(shells[s]))
tempj2.write('\n')
tempt2.write('\n')
for s in range(num_shells):
 jsh = j_sh[s] * (j_sh[s] + 1) 
 tempj2.write(' %f ' % jsh )     #One body term of J^2 equal to j(j+1) of shell
 tempt2.write(' %f ' % 0.75 )    #One body term of T^2 equal to 1/2(1/2+1)
tempj2.write('\n')
tempt2.write('\n')

tmin=0
tmax=1
for i in listtwoshells: 
 ja = float(int(i[0][-1])/2.0)
 jb = float(int(i[1][-1])/2.0)
 jmin, jmax = abs(ja-jb), abs(ja+jb)
 jvals = range(int(jmin),int(jmax)+1,1) 
 tempj2.write('%2i %i %2i %2i %2i %2i %i %i \n'  % (tmin, tmax, int(i[0]), int(i[1]), int(i[0]), int(i[1]), jmin, jmax) )
 tempt2.write('%2i %i %2i %2i %2i %2i %i %i \n'  % (tmin, tmax, int(i[0]), int(i[1]), int(i[0]), int(i[1]), jmin, jmax) )
 for t in [tmin,tmax]:
  for j in jvals:
   tpj = t + j
   if ( tpj % 2 == 0 and i[0] == i[1] ):
    jtmp = 0.0
    ttmp = 0.0
   else: 
    #Two body term equal to the coupled J or T minus the one-body term of shells a and b
    jtmp = j * (j + 1) - ja * (ja + 1) - jb * (jb + 1)
    ttmp = t * (t + 1) - 0.5 * (0.5 + 1) - 0.5 * (0.5 + 1)
   tempj2.write(' %f '  %  jtmp )
   tempt2.write(' %f '  %  ttmp )
  tempj2.write('\n')
  tempt2.write('\n')
  
tempt2.close()
tempj2.close()






print('---------------------------------------------------------------------')
print('J^2 AND T^2 FILES IN NUSHELLX FORMAT WRITTEN IN j2.int and t2.int')
print('---------------------------------------------------------------------')








