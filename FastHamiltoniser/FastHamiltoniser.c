#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "HamiltonianObject.c" //I know this is unusual, but we want the type definitions to be static


static PyModuleDef FastHamiltoniser = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "FastHamiltoniser",
    .m_doc = "Extension module to help rapidly compute hamiltonian simulations.",
    .m_size = -1,
};


PyMODINIT_FUNC PyInit_FastHamiltoniser(void) {
    import_array();

    PyObject* m;
    if (PyType_Ready(&HamiltonianMatrixType) < 0)
        return NULL;

    m = PyModule_Create(&FastHamiltoniser);
    if (m == NULL)
        return NULL;

    Py_INCREF(&HamiltonianMatrixType);
    if (PyModule_AddObject(m, "HamiltonianMatrix", (PyObject*)&HamiltonianMatrixType) < 0) {
        Py_DECREF(&HamiltonianMatrixType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}