import numpy as np
import FastHamiltoniser as m
hm = m.HamiltonianMatrix(dimension = 5)
print("trying to add permuter")
print(hm.add_permuter(np.array([1,2,3],dtype = np.int32),   np.array([2,3,4],dtype = np.int32),   np.array([1,-1,1],dtype = complex),  2j))
print(hm.add_permuter(np.array([1,2,3],dtype = np.int32),np.array([2,3,4],dtype = np.int32),np.array([1,-1,1],dtype = complex)))
print(hm.matrix())
hm.set_permuter_params(1,4)
print(hm.matrix())
print("trying rearrange vector on cpu")
j = np.array([5,4,3,2,1], dtype = complex)
print(hm.apply_cpu(j))
print("and now on the gpu")
print(hm.apply(j))

print("done")