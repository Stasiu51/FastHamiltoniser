#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>
#include "HamiltonianMatrix.h"

extern void apply_hamiltonian_gpu(HamiltonianMatrix* ham, _Dcomplex* vector, _Dcomplex* output_vector);

_Dcomplex _Caddcc(_Dcomplex c1, _Dcomplex c2) { return _Cbuild(creal(c1) + creal(c2), cimag(c1) + cimag(c2)); }

typedef struct {
    PyObject_HEAD
        HamiltonianMatrix* obj;
} PythonHamiltonianMatrix;


void apply_hamiltonian_cpu(PythonHamiltonianMatrix* self, _Dcomplex* vector, _Dcomplex* output_vector) {
    for (int p = 0; p < self->obj->n_permuters; p++) {

        Permuter* permuter = self->obj->permuters + p;
        if (!permuter->active) continue;
        for (int i = 0; i < permuter->size; i++) {
            output_vector[permuter->targets[i]] = _Cmulcc(_Cmulcc(permuter->coefficient, vector[permuter->sources[i]]), permuter->relative_couplings[i]);
        }
    }
}


static int
HamiltonianMatrix_init(PythonHamiltonianMatrix* self, PyObject* args, PyObject* kwds)
{
    int dim;
    static char* kwlist[] = { "dimension", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &dim))
        return NULL;

    self->obj = (HamiltonianMatrix*)malloc(sizeof(HamiltonianMatrix));
    if (self->obj == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate hamiltonian matrix object.");
        return NULL;
    }
    self->obj->dim = dim;
    self->obj->permuters = (Permuter*)malloc(sizeof(Permuter) * 2 * self->obj->dim);
    if (self->obj->permuters == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate permuter array.");
        return NULL;
    }

    
    
    self->obj->n_permuters = 0;
    return 0;
}

static PyObject* HamiltonianMatrix_getMatrix(PythonHamiltonianMatrix* self, PyObject* Py_UNUSED(ignored))
{
    npy_intp dims[2] = {self->obj->dim, self->obj->dim};
    PyArrayObject* ar = PyArray_Zeros(2, dims, PyArray_DescrFromType(NPY_COMPLEX128), 0);
    if (ar == NULL) {
        PyErr_SetString(PyExc_ValueError, "failed to make array.");
        return NULL;
    }
    _Dcomplex* c_data = (_Dcomplex*)ar->data;
    for (int p_i = 0; p_i < self->obj->n_permuters; p_i++) {
        Permuter* permuter = self->obj->permuters + p_i;
        if (!permuter->active) continue;
        for (int i = 0; i < permuter->size; i++) {
            int index = permuter->targets[i] * ar->dimensions[0] + permuter->sources[i];
            c_data[index] = _Caddcc(c_data[index], _Cmulcc(permuter->relative_couplings[i], permuter->coefficient));
        }
    }
    return (PyObject*) ar;
}

static PyObject* HamltonianMatrix_addPermuter(PythonHamiltonianMatrix* self, PyObject* args, PyObject* kwds) {
    PyArrayObject* sources;
    PyArrayObject* targets;
    PyArrayObject* relative_coupling_array;
    Py_complex coefficient = (Py_complex) { 1.0, 0 };
    int active = 1;
    static char* kwlist[] = { "target_array","source_array","relative_coupling_array","intial_coefficient","active", NULL};
    if (!PyArg_ParseTuple(args, "O!O!O!|Di", &PyArray_Type, &targets, &PyArray_Type, &sources, &PyArray_Type, &relative_coupling_array, &coefficient,&active))
        return NULL;
    if (sources->descr->type_num != PyArray_INT32 || sources->nd != 1) {
        PyErr_SetString(PyExc_ValueError, "The sources array must have type int32 and have dimension 1.");
        return NULL;
    }

    int size = sources->dimensions[0];
    if (size < 1) {
        PyErr_SetString(PyExc_ValueError, "The permuter must permute at least 1 element.");
        return NULL;
    }

    if (targets->nd != 1 || targets->descr->type_num != PyArray_INT32 || targets->dimensions[0] != size) {
        PyErr_SetString(PyExc_ValueError, "The targets array must be of type int32, be one-dimensional and have the same length as the sources array.");
        return NULL;
    }

    if (relative_coupling_array->nd != 1 || relative_coupling_array->descr->type_num != PyArray_COMPLEX128 || relative_coupling_array->dimensions[0] != size) {
        PyErr_SetString(PyExc_ValueError, "The relative coupling array must be of type complex, be one-dimensional and have the same length as the sources array.");
        return NULL;
    }

    Permuter* newPermuter = (Permuter*) malloc(sizeof(Permuter));
    if (newPermuter == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate permuter memory.");
        return NULL;
    }

    for (int i = 0; i < size; i++) {
        int s = ((int*)sources->data)[i];
        int t = ((int*)targets->data)[i];
        if (s < 0 || s >= self->obj->dim || t < 0 || t >= self->obj->dim) {
            PyErr_SetString(PyExc_ValueError, "All entries in the source and target arrays must be between 0 and (matrix dimension) - 1.");
            return NULL;
        }
    }
    newPermuter->active = active;
    newPermuter->coefficient = (_Dcomplex){ coefficient.real,coefficient.imag };
    newPermuter->sources = sources->data;
    newPermuter->targets = targets->data;
    newPermuter->relative_couplings = relative_coupling_array->data;
    newPermuter->size = size;
    Py_INCREF(sources); Py_INCREF(targets); Py_INCREF(relative_coupling_array);
    if (self->obj->n_permuters == self->obj->dim) {
        PyErr_SetString(PyExc_RuntimeError, "Maximum number of permuters added = 2 x dimension of matrix.");
        return NULL;
    }
    self->obj->permuters[self->obj->n_permuters] = *newPermuter;
    self->obj->n_permuters++;
    Py_BuildValue("N", MyPyLong_FromInt64((long long) self->obj->n_permuters - 1));
}

static PyObject* HamiltonianMatrix_setPermuterState(PythonHamiltonianMatrix* self, PyObject* args, PyObject* kwds) {
    int p_i;
    Py_complex coefficient;
    int active = NULL;

    static char* kwlist[] = { "permuter_index","coefficient","active", NULL };
    if (!PyArg_ParseTuple(args, "iD|i", &p_i, &coefficient,&active)) {
        return NULL;
    }
    if (p_i < 0 || p_i >= self->obj->n_permuters) {
        PyErr_Format(PyExc_ValueError, "The index must be between 0 and %i (the number of permuters added - 1)", self->obj->n_permuters - 1);
        return NULL;
    }
    Permuter* permuter = self->obj->permuters + p_i;
    permuter->coefficient = (_Dcomplex){ coefficient.real,coefficient.imag };
    if (active == NULL) {
        if (cabs(permuter->coefficient) < 1e-30) active = 0;
        else active = 1;
    }
    permuter->active = active;

    Py_RETURN_NONE;
}

static PyObject* HamiltonianMatrix_applyToVector_cpu(PythonHamiltonianMatrix* self, PyObject* args) {
    PyArrayObject* vector;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &vector)) {
        return NULL;
    }
    if (vector->nd != 1 || vector->descr->type_num != PyArray_COMPLEX128 || vector->dimensions[0] != self->obj->dim) {
        PyErr_SetString(PyExc_ValueError, "Vector array must be one dimension, of type complex and of length = dimension of the matrix.");
        return NULL;
    }
    _Dcomplex* output_vector = malloc(sizeof(_Dcomplex)*self->obj->dim);
    if (output_vector == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output vector.");
        return NULL;
    }
    apply_hamiltonian_cpu(self, (_Dcomplex*) vector->data, output_vector);
    npy_intp dims[1] = { self->obj->dim };
    return PyArray_SimpleNewFromData(1, dims, PyArray_COMPLEX128, (Py_complex*) output_vector);
}

static PyObject* HamiltonianMatrix_applyToVector_gpu(PythonHamiltonianMatrix* self, PyObject* args) {
    PyArrayObject* vector;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &vector)) {
        return NULL;
    }
    if (vector->nd != 1 || vector->descr->type_num != PyArray_COMPLEX128 || vector->dimensions[0] != self->obj->dim) {
        PyErr_SetString(PyExc_ValueError, "Vector array must be one dimension, of type complex and of length = dimension of the matrix.");
        return NULL;
    }
    _Dcomplex* output_vector = malloc(sizeof(_Dcomplex) * self->obj->dim);
    if (output_vector == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output vector.");
        return NULL;
    }
    apply_hamiltonian_gpu(self->obj, (_Dcomplex*)vector->data, output_vector);
    npy_intp dims[1] = { self->obj->dim };
    return PyArray_SimpleNewFromData(1, dims, PyArray_COMPLEX128, (Py_complex*)output_vector);
}

static PyMethodDef HamiltonianMatrix_methods[] = {
    {"matrix", HamiltonianMatrix_getMatrix, METH_NOARGS,
     "Return the matrix form of the hamiltonian as a numpy array. For performance, use 'apply' instead."},
    {"add_permuter",HamltonianMatrix_addPermuter,METH_VARARGS,
    "Add a permuter step to the hamiltonian matrix. Returns the permuter index."},
    {"set_permuter_params",HamiltonianMatrix_setPermuterState,METH_VARARGS,
    "Adjust the coefficient (and active state) of an existing permuter."},
    {"apply_cpu", HamiltonianMatrix_applyToVector_cpu, METH_VARARGS,
    "Apply the hamiltonian matrix to a complex vector on the cpu and return a modified copy."},
    {"apply", HamiltonianMatrix_applyToVector_gpu, METH_VARARGS,
    "Apply the hamiltonian matrix to a complex vector and return a modified copy."},
    {NULL}  /* Sentinel */
};

static PyTypeObject HamiltonianMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "FastHamiltoniser.HamiltonianMatrix",
    .tp_basicsize = sizeof(PythonHamiltonianMatrix),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Custom objects"),
    .tp_new = PyType_GenericNew,
    .tp_init = HamiltonianMatrix_init,
    .tp_methods = HamiltonianMatrix_methods,
};