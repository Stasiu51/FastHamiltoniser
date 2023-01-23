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
