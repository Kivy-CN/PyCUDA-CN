Built-in Utilities
==================

Automatic Initialization
------------------------

The module pycuda.autoinit, when imported, automatically performs all
the steps necessary to get CUDA ready for submission of compute kernels.
It uses pycuda.tools.make\_default\_context to create a compute context.

Choice of Device
----------------

Kernel Caching
--------------

Testing
-------

Device Metadata and Occupancy
-----------------------------

Memory Pools
------------

The functions pycuda.driver.mem\_alloc and
pycuda.driver.pagelocked\_empty can consume a fairly large amount of
processing time if they are invoked very frequently. For example, code
based on pycuda.gpuarray.GPUArray can easily run into this issue because
a fresh memory area is allocated for each intermediate result. Memory
pools are a remedy for this problem based on the observation that often
many of the block allocations are of the same sizes as previously used
ones.

Then, instead of fully returning the memory to the system and incurring
the associated reallocation overhead, the pool holds on to the memory
and uses it to satisfy future allocations of similarly-sized blocks. The
pool reacts appropriately to out-of-memory conditions as long as all
memory allocations are made through it. Allocations performed from
outside of the pool may run into spurious out-of-memory conditions due
to the pool owning much or all of the available memory.

### Device-based Memory Pool

### Memory Pool for pagelocked memory
