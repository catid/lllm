#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "tokenized_data_loader.hpp"
#include "tokenized_data_prep.hpp"

#include <Python.h>
#include <numpy/arrayobject.h> // Include NumPy array object

extern "C" {

//------------------------------------------------------------------------------
// Data Loader

static PyObject* data_loader_create(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    const char* index_file;
    uint32_t rank;
    uint32_t local_ranks;

    // Parse the arguments from Python
    if (!PyArg_ParseTuple(args, "sII", &index_file, &rank, &local_ranks)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid arguments for data_loader_create");
        return NULL; 
    }

    TokenizedDataLoader* loader = new TokenizedDataLoader();
    if (!loader->Start(index_file, rank, local_ranks)) {
        delete loader;
        PyErr_SetString(PyExc_RuntimeError, "Failed to start data loader");
        return NULL;
    }

    // Return a Python object representing the data loader (e.g., a PyLong object)
    return PyLong_FromVoidPtr(loader);
}

static PyObject* data_loader_destroy(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    void* data_loader;

    // Parse the arguments from Python
    if (!PyArg_ParseTuple(args, "K", &data_loader)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid arguments for data_loader_destroy");
        return NULL; 
    }

    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    if (!loader) {
        PyErr_SetString(PyExc_RuntimeError, "Data loader is NULL");
        return NULL;
    }
    loader->Stop();
    delete loader;

    Py_RETURN_NONE; // Indicate success with no return value
}

static PyObject* data_loader_start_epoch(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    void* data_loader;
    uint64_t seed0;
    uint64_t seed1;
    uint32_t micro_batch_size;
    uint32_t context_size;

    // Parse the arguments from Python
    if (!PyArg_ParseTuple(args, "KLLII", &data_loader, &seed0, &seed1, &micro_batch_size, &context_size)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid arguments for data_loader_start_epoch");
        return NULL; 
    }

    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    loader->StartEpoch(
        seed0, seed1,
        micro_batch_size,
        context_size);

    Py_RETURN_NONE; // Indicate success with no return value
}

static PyObject* data_loader_get_micro_batch(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    void* data_loader;
    uint32_t micro_batch_size;
    uint32_t num_tokens;
    PyObject* is_continuation;

    // Parse the arguments from Python
    if (!PyArg_ParseTuple(args, "KI", &data_loader, &micro_batch_size)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid arguments for data_loader_get_micro_batch");
        return NULL; 
    }

    // Initialize NumPy
    import_array(); 

    // Create a NumPy array to store the output batch
    npy_intp dims = {static_cast<npy_intp>(num_tokens)};
    PyObject* output_batch = PyArray_SimpleNew(1, &dims, NPY_UINT32);
    if (!output_batch) {
        // Handle error creating NumPy array
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array for output batch");
        return NULL;
    }

    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    uint8_t is_cont = 0;
    bool success = loader->GetTokenArray(
        &micro_batch_size,
        &num_tokens,
        static_cast<uint32_t*>(PyArray_DATA((PyArrayObject*)output_batch)),
        &is_cont);

    if (!success) {
        // Handle failure to get micro batch
        Py_DECREF(output_batch);
        PyErr_SetString(PyExc_RuntimeError, "Failed to get micro batch");
        return NULL; 
    }

    // Create a Python bool object for is_continuation
    is_continuation = PyBool_FromLong(is_cont);

    // Return the tuple containing the results
    return Py_BuildValue("(iOO)", micro_batch_size, output_batch, is_continuation);
}

//------------------------------------------------------------------------------
// Data Preparation

static PyObject* data_prep_create(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    const char* data_folder_path;

    // Parse the argument from Python
    if (!PyArg_ParseTuple(args, "s", &data_folder_path)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid argument for data_prep_create");
        return NULL; 
    }

    TokenizedDataPrep* prep = new TokenizedDataPrep();
    prep->Start(data_folder_path);

    // Return a Python object representing the data preparation object (e.g., a PyLong object)
    return PyLong_FromVoidPtr(prep);
}

static PyObject* data_prep_destroy(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    void* data_prep;

    // Parse the arguments from Python
    if (!PyArg_ParseTuple(args, "K", &data_prep)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid arguments for data_prep_destroy");
        return NULL; 
    }

    delete static_cast<TokenizedDataPrep*>(data_prep);
    Py_RETURN_NONE; // Indicate success with no return value
}

static PyObject* data_prep_write_tokenized_text(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    void* data_prep;
    PyObject* tokenized_text_obj;
    uint32_t text_length;

    // Parse the arguments from Python
    if (!PyArg_ParseTuple(args, "OKI", &data_prep, &tokenized_text_obj, &text_length)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid arguments for data_prep_write_tokenized_text");
        return NULL; 
    }

    // Convert Python object to C++ array
    const uint32_t* tokenized_text = static_cast<const uint32_t*>(PyArray_DATA((PyArrayObject*)tokenized_text_obj));
    if (!tokenized_text) {
        // Handle error converting Python object to C++ array
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert Python object to C++ array");
        return NULL; 
    }

    TokenizedDataPrep* prep = static_cast<TokenizedDataPrep*>(data_prep);
    bool success = prep->WriteTokenizedText(tokenized_text, text_length);

    if (!success) {
        // Handle failure to write tokenized text
        PyErr_SetString(PyExc_RuntimeError, "Failed to write tokenized text");
        return NULL; 
    }

    Py_RETURN_NONE; // Indicate success with no return value
}

//------------------------------------------------------------------------------
// Data Verification

static PyObject* data_verify(PyObject* self, PyObject* args) {
    (void)self; // Suppress unused parameter warning
    const char* data_folder_path;

    // Parse the argument from Python
    if (!PyArg_ParseTuple(args, "s", &data_folder_path)) {
        // Handle parsing error (e.g., raise a Python exception)
        PyErr_SetString(PyExc_TypeError, "Invalid argument for data_verify");
        return NULL; 
    }

    bool success = VerifyDataset(data_folder_path);

    if (!success) {
        // Handle failure to verify dataset
        PyErr_SetString(PyExc_RuntimeError, "Failed to verify dataset");
        return NULL; 
    }

    Py_RETURN_TRUE; // Indicate success
}

//------------------------------------------------------------------------------
// Python Bindings

static PyMethodDef cpp_dataloader_methods[] = {
    // Data Loader functions
    {"data_loader_create", (PyCFunction)data_loader_create, METH_VARARGS, "Create a data loader"},
    {"data_loader_destroy", (PyCFunction)data_loader_destroy, METH_VARARGS, "Destroy a data loader"},
    {"data_loader_start_epoch", (PyCFunction)data_loader_start_epoch, METH_VARARGS, "Start an epoch in the data loader"},
    {"data_loader_get_micro_batch", (PyCFunction)data_loader_get_micro_batch, METH_VARARGS, "Get the next micro batch"},

    // Data Preparation functions
    {"data_prep_create", (PyCFunction)data_prep_create, METH_VARARGS, "Create a data preparation object"},
    {"data_prep_destroy", (PyCFunction)data_prep_destroy, METH_VARARGS, "Destroy a data preparation object"},
    {"data_prep_write_tokenized_text", (PyCFunction)data_prep_write_tokenized_text, METH_VARARGS, "Write tokenized text to a file"},

    // Data Verification functions
    {"data_verify", (PyCFunction)data_verify, METH_VARARGS, "Verify the integrity of the dataset"},

    {NULL, NULL, 0, NULL} // Sentinel value
};

static struct PyModuleDef m_cpp_dataloader_module = {
    PyModuleDef_HEAD_INIT,
    "cpp_dataloader",  // Name of the module
    "C++ Data Loader and Preparation Module",  // Module documentation
    -1,
    cpp_dataloader_methods,
    NULL, // m_slots
    NULL, // m_traverse
    NULL, // m_clear
    NULL  // m_free
};

PyMODINIT_FUNC PyInit_cpp_dataloader(void) {
    return PyModule_Create(&m_cpp_dataloader_module); 
}

} // extern "C"
