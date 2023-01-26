#pragma once
#include<complex.h>

typedef struct {
    int active; //Is this map active (active=0 produces the same result as coefficient = 0)
    int size; // the number of elements to be permuted.
    _Dcomplex coefficient; //Coefficient of the map
    int* sources;
    int* targets;
    _Dcomplex* relative_couplings; // what the contributing element is multiplied by before adding to the target element.
} Permuter;

typedef struct {
    int dim;
    Permuter* permuters;
    int n_permuters;
} HamiltonianMatrix;

