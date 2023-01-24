#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>

typedef struct Permuters {
    int active; //Is this map active (active=0 produces the same result as coefficient = 0)
    int size; // the number of elements to be permuted.
    _Dcomplex coefficient; //Coefficient of the map
    int* sources;
    int* targets; //mapping[2] = 3 implies that the value of the 4th element contributes to the value of the 3rd (target) element.
    _Dcomplex* relative_couplings; // what the contributing element is multiplied by before adding to the target element.
} Permuter;

typedef struct {
    PyObject_HEAD
    int dim;
    Permuter* permuters;
    int n_permuters;
} HamiltonianMatrix;


//static PyObject* HamiltonianMatrix_new(PyTypeObject *type, PyObject* args, PyObject* kwds)
//{
//    printf("hello!\n");
//    
//    HamiltonianMatrix* self;
//    self = (HamiltonianMatrix*)type->tp_alloc(type, 0);
//    if (self != NULL) {
//        self->dim = 0;
//    }
//    return (PyObject*)self;
//}

static int
HamiltonianMatrix_init(HamiltonianMatrix* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = { "dimension", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &self->dim))
        return -1;
    self->permuters = (Permuter*)malloc(sizeof(Permuter) * 2*self->dim);
    if (self->permuters == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate permuter array.");
        return NULL;
    }
    self->n_permuters = 0;
    return 0;
}

static PyObject* HamiltonianMatrix_getMatrix(HamiltonianMatrix* self, PyObject* args)
{
    npy_intp dims[2] = {self->dim, self->dim};
    PyArrayObject* ar = PyArray_Zeros(2, dims, PyArray_DescrFromType(NPY_DOUBLE), 0);
    if (ar == NULL) {
        PyErr_SetString(PyExc_ValueError, "failed to make array.");
        return NULL;
    }
    return ar;
}

static PyObject* HamltonianMatrix_addPermuter(HamiltonianMatrix* self, PyObject* args, PyObject* kwds) {
    PyArrayObject* sources;
    PyArrayObject* targets;
    PyArrayObject* relative_coupling_array;
    Py_complex* coefficient;
    int active = 1;
    static char* kwlist[] = { "source_array","target_array","relative_coupling_array","intial_coefficient","active", NULL};
    if (!PyArg_ParseTuple(args, "O!O!O!|Di", &PyArray_Type, &sources, &PyArray_Type, &targets, &PyArray_Type, &relative_coupling_array, &coefficient,&active))
        return NULL;
    if (sources->descr->type_num != PyArray_INT16 || sources->nd != 1) {
        PyErr_SetString(PyExc_ValueError, "The sources array must have type int16 and have dimension 1.");
        return NULL;
    }
    int size = sources->dimensions[0];
    if (size < 1 || size > self->dim) {
        PyErr_SetString(PyExc_ValueError, "The permuter must permute between 1 and %i (the matrix dimension) elements.", self->dim);
        return NULL;
    }
    if (targets->nd != 1 || targets->descr->type_num != PyArray_INT16 || targets->dimensions[0] != size) {
        PyErr_SetString(PyExc_ValueError, "The targets array must be of type int16, be one-dimensional and have the same length as the sources array.", self->dim);
        return NULL;
    }
    if (relative_coupling_array->nd != 1 || relative_coupling_array->descr->type_num != PyArray_COMPLEX128 || relative_coupling_array->dimensions[0] != size) {
        PyErr_SetString(PyExc_ValueError, "The relative coupling array must be of type complex, be one-dimensional and have the same length as the sources array.", self->dim);
        return NULL;
    }
    Permuter* newPermuter = malloc(sizeof(Permuter));
    if (newPermuter == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate permuter memory.");
        return NULL;
    }
    newPermuter->active = active;
    newPermuter->coefficient = (_Dcomplex) { coefficient->real,coefficient->imag };
    newPermuter->sources = sources->data;
    newPermuter->targets = targets->data;
    newPermuter->relative_couplings = relative_coupling_array->data;
    Py_INCREF(sources); Py_INCREF(targets); Py_INCREF(relative_coupling_array);
    if (self->n_permuters == self->dim) {
        PyErr_SetString(PyExc_RuntimeError, "Maximum number of permuters added = 2xdimension of matrix.");
        return NULL;
    }
    self->permuters[self->n_permuters] = *newPermuter;
    self->n_permuters++;
    Py_RETURN_NONE;
}

//static PyObject* HamiltonianMatrix_applyToVector(HamiltonianMatrix* self, PyObject* args) {
//
//}

static PyMethodDef HamiltonianMatrix_methods[] = {
    {"matrix", HamiltonianMatrix_getMatrix, METH_VARARGS,
     "Return the matrix form of the hamiltonian as a numpy array."},
    {"add_permuter",HamltonianMatrix_addPermuter,METH_VARARGS,
    "Add a permuter step to the hamiltonian matrix."},
    {NULL}  /* Sentinel */
};

static PyTypeObject HamiltonianMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "FastHamiltoniser.HamiltonianMatrix",
    .tp_basicsize = sizeof(HamiltonianMatrix),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Custom objects"),
    .tp_new = PyType_GenericNew,
    .tp_init = HamiltonianMatrix_init,
    .tp_methods = HamiltonianMatrix_methods,
};