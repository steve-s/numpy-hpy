#ifndef NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_

NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp);

NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp);

NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj);

NPY_NO_EXPORT int
array_items_visit(PyArrayObject_fields *fa, HPyFunc_visitproc visit, void *arg);

NPY_NO_EXPORT int
array_clear_hpyfields(HPyContext *ctx, HPy h_arr);

NPY_NO_EXPORT int
array_clear_hpyfields_Item(HPyContext *ctx, HPy h_arr, char *data, PyArray_Descr *descr);

NPY_NO_EXPORT int
array_fixup_hpyfields(HPyContext *ctx, HPy h_src, HPy h_dst);

NPY_NO_EXPORT int
array_fixup_hpyfields_Item(HPyContext *ctx, HPy h_src, HPy h_dest,
        char *data, PyArray_Descr *descr);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_ */
