Device Interface
================

Version Queries
---------------

Error Reporting
---------------

Constants
---------

### Graphics-related constants

Devices and Contexts
--------------------

Concurrency and Streams
-----------------------

Memory
------

### Global Device Memory

### Pagelocked Host Memory

#### Pagelocked Allocation

The numpy.ndarray instances returned by these functions have an
attribute *base* that references an object of type

#### Aligned Host Memory

The numpy.ndarray instances returned by these functions have an
attribute *base* that references an object of type

#### Post-Allocation Pagelocking

### Managed Memory

CUDA 6.0 adds support for a "Unified Memory" model, which creates a
managed virtual memory space that is visible to both CPUs and GPUs. The
OS will migrate the physical pages associated with managed memory
between the CPU and GPU as needed. This allows a numpy array on the host
to be passed to kernels without first creating a DeviceAllocation and
manually copying the host data to and from the device.

> **note**
>
> Managed memory is only available for some combinations of CUDA device,
> operating system, and host compiler target architecture. Check the
> CUDA C Programming Guide and CUDA release notes for details.

> **warning**
>
> This interface to managed memory should be considered experimental. It
> is provided as a preview, but for now the same interface stability
> guarantees as for the rest of PyCUDA do not apply.

#### Managed Memory Allocation

The numpy.ndarray instances returned by these functions have an
attribute *base* that references an object of type

#### Managed Memory Usage

A managed numpy array is constructed and used on the host in a similar
manner to a pagelocked array:

    from pycuda.autoinit import context
    import pycuda.driver as cuda
    import numpy as np

    a = cuda.managed_empty(shape=10, dtype=np.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    a[:] = np.linspace(0, 9, len(a)) # Fill array on host

It can be passed to a GPU kernel, and used again on the host without an
explicit copy:

    from pycuda.compiler import SourceModule
    mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        a[threadIdx.x] *= 2;
    }
    """)
    doublify = mod.get_function("doublify")

    doublify(a, grid=(1,1), block=(len(a),1,1))
    context.synchronize() # Wait for kernel completion before host access

    median = np.median(a) # Computed on host!

> **warning**
>
> The CUDA Unified Memory model has very specific rules regarding
> concurrent access of managed memory allocations. Host access to any
> managed array is not allowed while the GPU is executing a kernel,
> regardless of whether the array is in use by the running kernel.
> Failure to follow the concurrency rules will generate a segmentation
> fault, *causing the Python interpreter to terminate immediately*.
>
> Users of managed numpy arrays should read the "Unified Memory
> Programming" appendix of the CUDA C Programming Guide for further
> details on the concurrency restrictions.
>
> If you are encountering interpreter terminations due to concurrency
> issues, the faulthandler \<http://pypi.python.org/pypi/faulthandler\>
> module may be helpful in locating the location in your Python program
> where the faulty access is occurring.

### Arrays and Textures

### Initializing Device Memory

### Unstructured Memory Transfers

### Structured Memory Transfers

Code on the Device: Modules and Functions
-----------------------------------------

Profiler Control
================

CUDA 4.0 and newer.

Just-in-time Compilation
========================
