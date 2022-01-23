/*
 * This module corresponds to the `Special functions for NPY_OBJECT`
 * section in the numpy reference for C-API.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "iterators.h"

#include "npy_config.h"

#include "npy_pycompat.h"

static void
HPyArray_FillObjectArray(HPyContext *ctx, HPy h_arr, HPy h_obj);

static void
_fillobject(HPyContext *ctx, HPy h_arr, char *data, HPy h_obj, PyArray_Descr *dtype);


/*NUMPY_API
 * XINCREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == NPY_OBJECT) {
        memcpy(&temp, data, sizeof(temp));
        Py_XINCREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(descr->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return;
            }
            PyArray_Item_INCREF(data + offset, new);
        }
    }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        for (i = 0; i < size; i++){
            /* Recursively increment the reference count of subarray elements */
            PyArray_Item_INCREF(data + i * inner_elsize,
                                descr->subarray->base);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}


/*NUMPY_API
 *
 * XDECREF all objects in a single array item. This is complicated for
 * structured datatypes where the position of objects needs to be extracted.
 * The function is execute recursively for each nested field or subarrays dtype
 * such as as `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    if (descr->type_num == NPY_OBJECT) {
        memcpy(&temp, data, sizeof(temp));
        Py_XDECREF(temp);
    }
    else if (PyDataType_HASFIELDS(descr)) {
            PyObject *key, *value, *title = NULL;
            PyArray_Descr *new;
            int offset;
            Py_ssize_t pos = 0;

            while (PyDict_Next(descr->fields, &pos, &key, &value)) {
                if (NPY_TITLE_KEY(key, value)) {
                    continue;
                }
                if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                      &title)) {
                    return;
                }
                PyArray_Item_XDECREF(data + offset, new);
            }
        }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        for (i = 0; i < size; i++){
            /* Recursively decrement the reference count of subarray elements */
            PyArray_Item_XDECREF(data + i * inner_elsize,
                                 descr->subarray->base);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}

/* Used for arrays of python objects to increment the reference count of */
/* every python object in the array. */
/*NUMPY_API
  For object arrays, increment all internal references.
*/
NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    PyArrayIterObject *it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            PyArray_Item_INCREF(it->dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                Py_XINCREF(*data);
            }
        }
        else {
            for( i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XINCREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            memcpy(&temp, it->dataptr, sizeof(temp));
            Py_XINCREF(temp);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
  Decrement all internal references for object arrays.
  (or arrays with object fields)
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    /*
     * statically allocating it allows this function to not modify the
     * reference count of the array for use during dealloc.
     * (statically is not necessary as such)
     */
    PyArrayIterObject it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            PyArray_Item_XDECREF(it.dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(&it);
        }
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) Py_XDECREF(*data);
        }
        else {
            for (i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XDECREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            memcpy(&temp, it.dataptr, sizeof(temp));
            Py_XDECREF(temp);
            PyArray_ITER_NEXT(&it);
        }
    }
    return 0;
}

static int
visit_item(char *data, PyArray_Descr *descr,
        HPyFunc_visitproc visit, void *arg)
{
    HPyField temp;

    if (!PyDataType_REFCHK(descr)) {
        return 0;
    }

    if (descr->type_num == NPY_OBJECT) {
        memcpy(&temp, data, sizeof(temp));
        HPy_VISIT(&temp);
        memcpy(data, &temp, sizeof(temp));
    }
    else if (PyDataType_HASFIELDS(descr)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(descr->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return 0;
            }
            visit_item(data + offset, new, visit, arg);
        }
    }
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        inner_elsize = descr->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return 0;
        }
        /* Subarrays are always contiguous in memory */
        size = descr->elsize / inner_elsize;

        for (i = 0; i < size; i++){
            visit_item(data + i * inner_elsize, descr->subarray->base, visit, arg);
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return 0;
}

NPY_NO_EXPORT int
array_items_visit(PyArrayObject_fields *fa, HPyFunc_visitproc visit, void *arg)
{
    npy_intp i, n;
    HPyField *data;
    PyArrayIterObject it;
    HPyField temp;
    PyArrayObject *mp = (PyArrayObject *)fa;
    PyArray_Descr *descr = PyArray_DESCR(mp);

    if (!PyDataType_REFCHK(descr)) {
        return 0;
    }

    if (descr->type_num != NPY_OBJECT) {
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            visit_item(it.dataptr, descr, visit, arg);
            PyArray_ITER_NEXT(&it);
        }
        return 0;
    }
    if (PyArray_ISONESEGMENT(mp)) {
        data = (HPyField *)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                HPy_VISIT(data);
            }
        }
        else {
            for (i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                HPy_VISIT(&temp);
                memcpy(data, &temp, sizeof(temp));
            }
        }
    }
    else { /* handles misaligned data too */
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            memcpy(&temp, it.dataptr, sizeof(temp));
            HPy_VISIT(&temp);
            memcpy(it.dataptr, &temp, sizeof(temp));
            PyArray_ITER_NEXT(&it);
        }
    }
    return 0;
}

typedef struct {
    HPyContext *ctx;
    HPy h_arr;
} clear_fields_args_t;

static int
clear_fields_visitor(HPyField *pf, void *arg)
{
    clear_fields_args_t *args = (clear_fields_args_t *)arg;
    HPyField_Store(args->ctx, args->h_arr, pf, HPy_NULL);
    return 0;
}

NPY_NO_EXPORT int
array_clear_hpyfields(HPyContext *ctx, HPy h_arr)
{
    PyArrayObject_fields *fa = HPyArray_AsFields(ctx, h_arr);
    clear_fields_args_t args = {ctx, h_arr};
    return array_items_visit(fa, clear_fields_visitor, &args);
}


typedef struct {
    HPyContext *ctx;
    HPy h_src;
    HPy h_dest;
} fixup_fields_args_t;

static int
fixup_fields_visitor(HPyField *pf, void *arg)
{
    fixup_fields_args_t *args = (fixup_fields_args_t *)arg;
    HPy item = HPyField_Load(args->ctx, args->h_src, *pf);
    *pf = HPyField_NULL;
    HPyField_Store(args->ctx, args->h_dest, pf, item);
    HPy_Close(args->ctx, item);
    return 0;
}

/* Replace fields that have been memcpy'ed from a source to a destination
 * array with a proper copy, with the owner set as the destination array.
 * Equivalent to PyArray_INCREF.
 */
NPY_NO_EXPORT int
array_fixup_hpyfields(HPyContext *ctx, HPy h_src, HPy h_dest)
{
    PyArrayObject_fields *fa = HPyArray_AsFields(ctx, h_dest);
    fixup_fields_args_t args = {ctx, h_src, h_dest};
    return array_items_visit(fa, fixup_fields_visitor, &args);
}


/*NUMPY_API
 * Assumes contiguous
 */
NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj)
{
    HPyContext *ctx = npy_get_context();
    HPy h_arr = HPy_FromPyObject(ctx, (PyObject *)arr);
    HPy h_obj = HPy_FromPyObject(ctx, obj);
    HPyArray_FillObjectArray(ctx, h_arr, h_obj);
    HPy_Close(ctx, h_obj);
    HPy_Close(ctx, h_arr);
}

static void
HPyArray_FillObjectArray(HPyContext *ctx, HPy h_arr, HPy h_obj)
{
    npy_intp i,n;
    PyArrayObject *arr = (PyArrayObject *)HPyArray_AsFields(ctx, h_arr);
    n = PyArray_SIZE(arr);
    if (PyArray_DESCR(arr)->type_num == NPY_OBJECT) {
        HPyField *fptr = (HPyField *)(PyArray_DATA(arr));
        n = PyArray_SIZE(arr);
        for (i = 0; i < n; i++) {
            HPyField_Store(ctx, h_arr, fptr++, h_obj);
        }
    }
    else {
        char *data = PyArray_DATA(arr);
        for (i = 0; i < n; i++) {
            _fillobject(ctx, h_arr, data, h_obj, PyArray_DESCR(arr));
            data += PyArray_DESCR(arr)->elsize;
        }
    }
}

static void
_fillobject(HPyContext *ctx, HPy h_arr, char *data, HPy h_obj, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        PyObject *obj = HPy_AsPyObject(ctx, h_obj);
        if ((obj == Py_None) ||
                (PyLong_Check(obj) && PyLong_AsLong(obj) == 0)) {
            return;
        }
        /* Clear possible long conversion error */
        PyErr_Clear();
        PyObject *arr = (PyObject *)HPyArray_AsFields(ctx, h_arr);
        if (arr!=NULL) {
            dtype->f->setitem(obj, data, arr);
        }
    }
    if (dtype->type_num == NPY_OBJECT) {
        HPyField f_item;
        memcpy(&f_item, data, sizeof(f_item));
        HPyField_Store(ctx, h_arr, &f_item, h_obj);
        memcpy(data, &f_item, sizeof(f_item));
    }
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _fillobject(ctx, h_arr, data + offset, h_obj, new);
        }
    }
    else if (PyDataType_HASSUBARRAY(dtype)) {
        int size, i, inner_elsize;

        inner_elsize = dtype->subarray->base->elsize;
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        /* Subarrays are always contiguous in memory */
        size = dtype->elsize / inner_elsize;

        /* Call _fillobject on each item recursively. */
        for (i = 0; i < size; i++){
            _fillobject(ctx, h_arr, data, h_obj, dtype->subarray->base);
            data += inner_elsize;
        }
    }
    else {
        /* This path should not be reachable. */
        assert(0);
    }
    return;
}
