#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>

typedef struct {
    PyObject_HEAD
    int dim;
} HamiltonianMatrix;

typedef struct Books {
    char title[50];
    char author[50];
    char subject[100];
    int book_id;
} Book;

typedef struct Permuters {
    int active;
    _Dcomplex coefficient;

} Permuter;


static PyObject* HamiltonianMatrix_new(PyTypeObject *type, PyObject* args, PyObject* kwds)
{
    printf("hello!\n");
    
    HamiltonianMatrix* self;
    self = (HamiltonianMatrix*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->dim = 10;
    }
    return (PyObject*)self;
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

static PyMethodDef HamiltonianMatrix_methods[] = {
    {"matrix", HamiltonianMatrix_getMatrix, METH_VARARGS,
     "Return the matrix form of the hamiltonian as a numpy array."},
    {NULL}  /* Sentinel */
};

static PyTypeObject HamiltonianMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "FastHamiltoniser.HamiltonianMatrix",
    .tp_basicsize = sizeof(HamiltonianMatrix),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Custom objects"),
    .tp_new = HamiltonianMatrix_new,
    .tp_methods = HamiltonianMatrix_methods,
};