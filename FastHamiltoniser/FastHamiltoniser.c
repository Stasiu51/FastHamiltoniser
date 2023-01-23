#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "HamiltonianObject.h"

extern double* vector_add_gpu(double* a, double* b, int num_elements);

static PyObject* helloworld(PyObject* self, PyObject* args) {
    printf("Hello World!\n");
    return Py_None;
}

static unsigned long long* memo = NULL; // static cache used to keep track of memoized values

unsigned long long cfib(int n) {
    unsigned long long value;

    if (n < 2 || memo[n] != 0)
        return memo[n];
    else {
        value = cfib(n - 1) + cfib(n - 2);
        memo[n] = value;
        return value;
    }
}

static PyObject* fib(PyObject* self, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;

    if (n == 6) {
        PyErr_SetString(PyExc_RuntimeError, "Not six!!");
        return NULL;
    }

    if (n < 2) {
        return Py_BuildValue("i", n);
    }

    memo = (unsigned long long*) calloc(n + 1, sizeof(unsigned long long));  // memoization, initialized to 0
    if (memo == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to dynamically allocate memory for memoization.");
        return NULL;
    }

    memo[0] = 0; // set initial conditions
    memo[1] = 1;

    // return our computed fib number
    PyObject* value = PyLong_FromUnsignedLongLong(cfib(n));
    free(memo);
    return Py_BuildValue("N", value);
}

static PyObject* vector_add(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;
    if (array1->nd != 1 || array2->nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }
    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    printf("running vector_add on dim1: %d, stride1: %d, dim2: %d, stride2: %d\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }
    double* output = (double*)malloc(sizeof(double) * n1);
    for (int i = 0; i < n1; i++)
        output[i] = *((double*)array1->data + i) + *((double*)array2->data + i);
    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}

static PyObject* vector_add_gpu_wrapper(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;
    double* output;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    if (array1->nd != 1 || array2->nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    printf("running vector_add on dim1: %Id, stride1: %Id, dim2: %Id, stride2: %Id\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }

    output = vector_add_gpu((double*)array1->data, (double*)array2->data, n1);

    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}

static PyMethodDef methods[] = {
    {"helloworld", helloworld, METH_NOARGS, "A Simple Hello World Function"}, // (function name, function, arguments, doc_string)
    {"fib", fib, METH_VARARGS, "Computes the nth Fibonacci number"}, // METH_VARARGS allows for arbitrary positional arguments
    {"vector_add",vector_add,METH_VARARGS,"add two numpy float vectors on the CPU."},
    {"vector_add_gpu",vector_add_gpu_wrapper,METH_VARARGS,"add two numpy float vectors on the GPU."},
    {NULL,NULL,0,NULL},
};

static PyModuleDef FastHamiltoniser = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "FastHamiltoniser",
    .m_doc = "Extension module to help rapidly compute hamiltonian simulations.",
    .m_size = -1,
    .m_methods = methods
};


PyMODINIT_FUNC PyInit_FastHamiltoniser(void) {
    import_array();

    PyObject* m;
    if (PyType_Ready(&CustomType) < 0)
        return NULL;

    m = PyModule_Create(&FastHamiltoniser);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CustomType);
    if (PyModule_AddObject(m, "Custom", (PyObject*)&CustomType) < 0) {
        Py_DECREF(&CustomType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}