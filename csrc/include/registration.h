#pragma once

#include <Python.h>

#define _IK_CONCAT(a, b) a##b
#define IK_CONCAT(a, b) _IK_CONCAT(a, b)
#define _IK_STRINGIFY(x) #x
#define IK_STRINGIFY(x) _IK_STRINGIFY(x)

#define REGISTER_EXTENSION(NAME)                                        \
  PyMODINIT_FUNC IK_CONCAT(PyInit_, NAME)() {                          \
    static struct PyModuleDef module = {                               \
        PyModuleDef_HEAD_INIT, IK_STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                   \
  }
