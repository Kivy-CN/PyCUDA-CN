Changes
=======

Version 2017.2
--------------

-   zeros\_like and empty\_like now have *dtype* and *order* arguments
    as in numpy. Previously these routines always returned a C-order
    array. The new default behavior follows the numpy default, which is
    to match the order and strides of the input as closely as possible.
-   A ones\_like gpuarray function was added.
-   methods GPUArray.imag, GPUArray.real, GPUArray.conj now all return
    Fortran-ordered arrays when the GPUArray is Fortran-ordered.

Version 2016.2
--------------

> **note**
>
> This version is the current development version. You can get it from
> [PyCUDA's version control
> repository](https://github.com/inducer/pycuda).

Version 2016.1
--------------

-   Bug fixes.
-   Global control of caching.
-   Matrix/array interop.
-   Add pycuda.gpuarray.GPUArray.squeeze

Version 2014.1
--------------

-   Add PointerHolderBase.as\_buffer and DeviceAllocation.as\_buffer.
-   Support for device\_attribute values added in CUDA 5.0, 5.5, and
    6.0.
-   Support for managed\_memory. (contributed by Stan Seibert)

Version 2013.1.1
----------------

-   Windows fix for PyCUDA on Python 3 (Thanks, Christoph Gohlke)

Version 2013.1
--------------

-   Python 3 support (large parts contributed by Tomasz Rybak)
-   Add pycuda.gpuarray.GPUArray.\_\_getitem\_\_, supporting generic
    slicing.

    It is *possible* to create non-contiguous arrays using this
    functionality. Most operations (elementwise etc.) will not work on
    such arrays.
-   More generators in pycuda.curandom. (contributed by Tomasz Rybak)
-   Many bug fixes

> **note**
>
> The addition of pyopencl.array.Array.\_\_getitem\_\_ has an unintended
> consequence due to [numpy bug
> 3375](https://github.com/numpy/numpy/issues/3375). For instance, this
> expression:
>
>     numpy.float32(5) * some_gpu_array
>
> may take a very long time to execute. This is because numpy first
> builds an object array of (compute-device) scalars (!) before it
> decides that that's probably not such a bright idea and finally calls
> pycuda.gpuarray.GPUArray.\_\_rmul\_\_.
>
> Note that only left arithmetic operations of pycuda.gpuarray.GPUArray
> by numpy scalars are affected. Python's number types (float etc.) are
> unaffected, as are right multiplications.
>
> If a program that used to run fast suddenly runs extremely slowly, it
> is likely that this bug is to blame.
>
> Here's what you can do:
>
> -   Use Python scalars instead of numpy scalars.
> -   Switch to right multiplications if possible.
> -   Use a patched numpy. See the bug report linked above for a pull
>     request with a fix.
> -   Switch to a fixed version of numpy when available.

Version 2012.1
--------------

-   Numerous bug fixes. (including shipped-boost compilation on gcc 4.7)

Version 2011.2
--------------

-   Fix a memory leak when using pagelocked memory. (reported by Paul
    Cazeaux)
-   Fix complex scalar argument passing.
-   Fix pycuda.gpuarray.zeros when used on complex arrays.
-   Add pycuda.tools.register\_dtype to enable scan/reduction on struct
    types.
-   More improvements to CURAND.
-   Add support for CUDA 4.1.

Version 2011.1.2
----------------

-   Various fixes.

Version 2011.1.1
----------------

-   Various fixes.

Version 2011.1
--------------

When you update code to run on this version of PyCUDA, please make sure
to have deprecation warnings enabled, so that you know when your code
needs updating. (See [the Python
docs](http://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x).
Caution: As of Python 2.7, deprecation warnings are disabled by
default.)

-   Add support for CUDA 3.0-style OpenGL interop. (thanks to Tomasz
    Rybak)
-   Add pycuda.driver.Stream.wait\_for\_event.
-   Add *range* and *slice* keyword argument to
    pycuda.elementwise.ElementwiseKernel.\_\_call\_\_.
-   Document *preamble* constructor keyword argument to
    pycuda.elementwise.ElementwiseKernel.
-   Add vector types, see pycuda.gpuarray.vec.
-   Add pycuda.scan.
-   Add support for new features in CUDA 4.0.
-   Add pycuda.gpuarray.GPUArray.strides,
    pycuda.gpuarray.GPUArray.flags. Allow the creation of arrys in C and
    Fortran order.
-   Adopt stateless launch interface from CUDA, deprecate old one.
-   Add CURAND wrapper. (with work by Tomasz Rybak)
-   Add pycuda.compiler.DEFAULT\_NVCC\_FLAGS.

Version 0.94.2
--------------

-   Fix the pesky Fermi reduction bug. (thanks to Tomasz Rybak)

Version 0.94.1
--------------

-   Support for CUDA debugging. (see
    [FAQ](http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions) for
    details.)

Version 0.94
------------

-   Support for CUDA 3.0. (but not CUDA 3.0 beta!) Search for "CUDA 3.0"
    in reference-doc to see what's new.
-   Support for CUDA 3.1 beta. Search for "CUDA 3.1" in reference-doc to
    see what's new.
-   Support for CUDA 3.2 RC. Search for "CUDA 3.2" in reference-doc to
    see what's new.
-   Add sparse matrix-vector multiplication and linear system solving
    code, in pycuda.sparse.
-   Add pycuda.gpuarray.if\_positive, pycuda.gpuarray.maximum,
    pycuda.gpuarray.minimum.
-   Deprecate pycuda.tools.get\_default\_device
-   Add pycuda.tools.make\_default\_context.
-   Use pycuda.tools.make\_default\_context in pycuda.autoinit, which
    changes its behavior.
-   Remove previously deprecated features:
    -   pycuda.driver.Function.registers, pycuda.driver.Function.lmem,
        and pycuda.driver.Function.smem have been deprecated in favor of
        the mechanism above. See pycuda.driver.Function.num\_regs for
        more.
    -   the three-argument forms (i.e. with streams) of
        pycuda.driver.memcpy\_dtoh and pycuda.driver.memcpy\_htod. Use
        pycuda.driver.memcpy\_dtoh\_async and
        pycuda.driver.memcpy\_htod\_async instead.
    -   pycuda.driver.SourceModule.
-   Add pycuda.tools.context\_dependent\_memoize, use it for
    context-dependent caching of PyCUDA's canned kernels.
-   Add pycuda.tools.mark\_cuda\_test.
-   Add attributes of pycuda.driver.CompileError. (requested by Dan
    Lepage)
-   Add preliminary support for complex numbers. (initial discussion
    with Daniel Fan)
-   Add pycuda.gpuarray.GPUArray.real, pycuda.gpuarray.GPUArray.imag,
    pycuda.gpuarray.GPUArray.conj.
-   Add pycuda.driver.PointerHolderBase.

Version 0.93
------------

> **warning**
>
> Version 0.93 makes some changes to the PyCUDA programming interface.
> In all cases where documented features were changed, the old usage
> continues to work, but results in a warning. It is recommended that
> you update your code to remove the warning.

-   OpenGL interoperability in pycuda.gl.
-   Document pycuda.gpuarray.GPUArray.\_\_len\_\_. Change its definition
    to match numpy.
-   Add pycuda.gpuarray.GPUArray.bind\_to\_texref\_ext.
-   Let pycuda.gpuarray.GPUArray operators deal with generic data types,
    including type promotion.
-   Add pycuda.gpuarray.take.
-   Fix thread handling by making internal context stack thread-local.
-   Add pycuda.reduction.ReductionKernel.
-   Add pycuda.gpuarray.sum, pycuda.gpuarray.dot,
    pycuda.gpuarray.subset\_dot.
-   Synchronous and asynchronous memory transfers are now separate from
    each other, the latter having an `_async` suffix. The
    now-synchronous forms still take a pycuda.driver.Stream argument,
    but this practice is deprecated and prints a warning.
-   pycuda.gpuarray.GPUArray no longer has an associated
    pycuda.driver.Stream. Asynchronous GPUArray transfers are now
    separate from synchronous ones and have an `_async` suffix.
-   Support for features added in CUDA 2.2.
-   pycuda.driver.SourceModule has been moved to
    pycuda.compiler.SourceModule. It is still available by the old name,
    but will print a warning about the impending deprecation.
-   pycuda.driver.Device.get\_attribute with a
    pycuda.driver.device\_attribute attr can now be spelled dev.attr,
    with no further namespace detours. (Suggested by Ian Cullinan)
    Likewise for pycuda.driver.Function.get\_attribute
-   pycuda.driver.Function.registers, pycuda.driver.Function.lmem, and
    pycuda.driver.Function.smem have been deprecated in favor of the
    mechanism above. See pycuda.driver.Function.num\_regs for more.
-   Add PyCUDA version query mechanism, see pycuda.VERSION.

Version 0.92
------------

> **note**
>
> If you're upgrading from prior versions, you may delete the directory
> \$HOME/.pycuda-compiler-cache to recover now-unused disk space.

> **note**
>
> During this release time frame, I had the honor of giving a talk on
> PyCUDA for a [class](http://sites.google.com/site/cudaiap2009/) that a
> group around Nicolas Pinto was teaching at MIT. If you're interested,
> the slides for it are
> [available](http://mathema.tician.de/dl/pub/pycuda-mit.pdf).

-   Make pycuda.tools.DeviceMemoryPool official functionality, after
    numerous improvements. Add pycuda.tools.PageLockedMemoryPool for
    pagelocked memory, too.
-   Properly deal with automatic cleanup in the face of several
    contexts.
-   Fix compilation on Python 2.4.
-   Fix 3D arrays. (Nicolas Pinto)
-   Improve error message when nvcc is not found.
-   Automatically run Python GC before throwing out-of-memory errors.
-   Allow explicit release of memory using
    pycuda.driver.DeviceAllocation.free,
    pycuda.driver.HostAllocation.free, pycuda.driver.Array.free,
    pycuda.tools.PooledDeviceAllocation.free,
    pycuda.tools.PooledHostAllocation.free.
-   Make configure switch `./configure.py --cuda-trace` to enable API
    tracing.
-   Add a documentation chapter and examples on metaprog.
-   Add pycuda.gpuarray.empty\_like and pycuda.gpuarray.zeros\_like.
-   Add and document pycuda.gpuarray.GPUArray.mem\_size in anticipation
    of stride/pitch support in pycuda.gpuarray.GPUArray.
-   Merge Jozef Vesely's MD5-based RNG.
-   Document pycuda.driver.from\_device and
    pycuda.driver.from\_device\_like.
-   Add pycuda.elementwise.ElementwiseKernel.
-   Various documentation improvements. (many of them from Nicholas
    Tung)
-   Move PyCUDA's compiler cache to the system temporary directory,
    rather than the users home directory.

Version 0.91
------------

-   Add support for compiling on CUDA 1.1. Added version query
    pycuda.driver.get\_version. Updated documentation to show 2.0-only
    functionality.
-   Support for Windows and MacOS X, in addition to Linux. (Gert
    Wohlgemuth, Cosmin Stejerean, Znah on the Nvidia forums, and David
    Gadling)
-   Support more arithmetic operators on pycuda.gpuarray.GPUArray. (Gert
    Wohlgemuth)
-   Add pycuda.gpuarray.arange. (Gert Wohlgemuth)
-   Add pycuda.curandom. (Gert Wohlgemuth)
-   Add pycuda.cumath. (Gert Wohlgemuth)
-   Add pycuda.autoinit.
-   Add pycuda.tools.
-   Add pycuda.tools.DeviceData and pycuda.tools.OccupancyRecord.
-   pycuda.gpuarray.GPUArray parallelizes properly on GTX200-generation
    devices.
-   Make pycuda.driver.Function resource usage available to the program.
    (See, e.g. pycuda.driver.Function.registers.)
-   Cache kernels compiled by pycuda.compiler.SourceModule. (Tom Annau)
-   Allow for faster, prepared kernel invocation. See
    pycuda.driver.Function.prepare.
-   Added memory pools, at pycuda.tools.DeviceMemoryPool as
    experimental, undocumented functionality. For some workloads, this
    can cure the slowness of pycuda.driver.mem\_alloc.
-   Fix the memset \<memset\> family of functions.
-   Improve errors.
-   Add order parameter to pycuda.driver.matrix\_to\_array and
    pycuda.driver.make\_multichannel\_2d\_array.

Acknowledgments
===============

-   Gert Wohlgemuth ported PyCUDA to MacOS X and contributed large parts
    of pycuda.gpuarray.GPUArray.
-   Alexander Mordvintsev contributed fixes for Windows XP.
-   Cosmin Stejerean provided multiple patches for PyCUDA's build
    system.
-   Tom Annau contributed an alternative SourceModule compiler cache as
    well as Windows build insight.
-   Nicholas Tung improved PyCUDA's documentation.
-   Jozef Vesely contributed a massively improved random number
    generator derived from the RSA Data Security, Inc. MD5 Message
    Digest Algorithm.
-   Chris Heuser provided a test cases for multi-threaded PyCUDA.
-   The reduction templating is based on code by Mark Harris at Nvidia.
-   Andrew Wagner provided a test case and contributed the port of the
    convolution example. The original convolution code is based on an
    example provided by Nvidia.
-   Hendrik Riedmann contributed the matrix transpose and list selection
    examples.
-   Peter Berrington contributed a working example for CUDA-OpenGL
    interoperability.
-   Maarten Breddels provided a patch for 'flat-egg' support.
-   Nicolas Pinto refactored pycuda.autoinit for automatic device
    finding.
-   Ian Ozsvald and Fabrizio Milo provided patches.
-   Min Ragan-Kelley solved the long-standing puzzle of why PyCUDA did
    not work on 64-bit CUDA on OS X (and provided a patch).
-   Tomasz Rybak solved another long-standing puzzle of why reduction
    failed to work on some Fermi chips. In addition, he provided a patch
    that updated PyCUDA's gl-interop to the state of CUDA 3.0.
-   Martin Bergtholdt of Philips Research provided a patch that made
    PyCUDA work on 64-bit Windows 7.

Licensing
=========

PyCUDA is licensed to you under the MIT/X Consortium license:

Copyright (c) 2009,10 Andreas Klöckner and Contributors.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

PyCUDA includes derivatives of parts of the
[Thrust](https://github.com/thrust/thrust/) computing package (in
particular the scan implementation). These parts are licensed as
follows:

> Copyright 2008-2011 NVIDIA Corporation
>
> Licensed under the Apache License, Version 2.0 (the "License"); you
> may not use this file except in compliance with the License. You may
> obtain a copy of the License at
>
> > \<<http://www.apache.org/licenses/LICENSE-2.0>\>
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
> implied. See the License for the specific language governing
> permissions and limitations under the License.

> **note**
>
> If you use Apache-licensed parts, be aware that these may be
> incompatible with software licensed exclusively under GPL2. (Most
> software is licensed as GPL2 or later, in which case this is not an
> issue.)

Frequently Asked Questions
==========================

The FAQ is now maintained collaboratively in the [PyCUDA
Wiki](http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions).

Citing PyCUDA
=============

We are not asking you to gratuitously cite PyCUDA in work that is
otherwise unrelated to software. That said, if you do discuss some of
the development aspects of your code and would like to highlight a few
of the ideas behind PyCUDA, feel free to cite [this
article](http://dx.doi.org/10.1016/j.parco.2011.09.001):

> Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul
> Ivanov, Ahmed Fasih, PyCUDA and PyOpenCL: A scripting-based approach
> to GPU run-time code generation, Parallel Computing, Volume 38, Issue
> 3, March 2012, Pages 157-174.

Here's a Bibtex entry for your convenience:

    @article{kloeckner_pycuda_2012,
       author = {{Kl{\"o}ckner}, Andreas
            and {Pinto}, Nicolas
            and {Lee}, Yunsup
            and {Catanzaro}, B.
            and {Ivanov}, Paul
            and {Fasih}, Ahmed },
       title = "{PyCUDA and PyOpenCL: A Scripting-Based Approach to GPU Run-Time Code Generation}",
       journal = "Parallel Computing",
       volume = "38",
       number = "3",
       pages = "157--174",
       year = "2012",
       issn = "0167-8191",
       doi = "10.1016/j.parco.2011.09.001",
    }
