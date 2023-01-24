import numpy as np
import FastHamiltoniser as m
print('Successfully imported. Running hello world:')
m.helloworld()
print('Running fib:')
print(m.fib(10))
a = np.array([1,2,3], dtype = float)
b = np.array([2,3,4], dtype = float)
print('Running cpu vector add:')
print(m.vector_add(a,b))
print('Running gpu vector add:')
print(m.vector_add_gpu(a,b))
print('Making custom object')
hm = m.HamiltonianMatrix(dimension = 5)
print(hm.matrix(a))

print("trying traverse")
e = np.array([[1,2,3],[4,5,6]], dtype = float)
m.iterate2Darray(e)


print("trying to add permuter")
hm.add_permuter(np.array([1,2,3],dtype = np.int16),np.array([2,3,4],dtype = np.int16),np.array([1,-1,1],dtype = complex))



print("done")