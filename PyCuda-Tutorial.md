Title: PyCUDA Tutorial 中文版
Date: 2016-10-13 11:20
Category: Python
Tags: Python,CUDA

[PyCUDA  Tutorial 英文原文](https://documen.tician.de/pycuda/)

[CycleUser](http://blog.cycleuser.org) 翻译

## 开始使用

在你使用PyCuda之前，要先用import命令来导入并初始化一下。




```Python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
```



这里要注意，你并不是**必须**使用pycuda.autoinit,初始化、内容的创建和清理也都可以手动实现。




## 转移数据

接下来就是要把数据转移到设备（device）上了。一般情况下，在使用PyCuda的时候，原始数据都是以NumPy数组的形式存储在宿主系统（host）中的。（不过实际上，只要符合Python缓冲区接口的数据类型就都可以使用的，甚至连字符串类型str都可以。）

（**译者注：宿主系统host，就是处理器-内存-外存组成的常规Python运行环境;设备device，就是你要拿来做CUDA运算的显卡或者运算卡，可以是单卡也可以是阵列。**）

下面这行示例代码创建了一个随机数组成的4*4大小的数组a：



```Python
import numpy
a = numpy.random.randn(4,4)
```



不过要先暂停一下—咱们刚刚创建的这个数组a包含的是双精度浮点数，但大多数常用的NVIDIA显卡只支持单精度浮点数，所以需要转换一下类型：

（**译者注：原作者的这篇简介主要针对使用常规普通显卡的用户，比如GeForce系列的各种大家平时用到的都是这个范围的，相比专门的计算卡，在双精度浮点数等方面进行了阉割，所以作者才建议转换类型到单精度浮点数。如果你使用的是专门的计算卡，就不用这样了。**）

```Python
a = a.astype(numpy.float32)
```





接下来，要把已有的数据转移过去，还要设定一个目的地，所以我们要在显卡中分配一段显存：

（**译者注：原文说的是设备，这里就直接说成显卡了，毕竟大家用显卡的比较多。另外下面这个代码中的a.nbytes是刚刚生成的数组a的大小，这里作者是按照数组大小来分配的显存，新入门的用户要注意这里，后续的使用中，显存的高效利用是很重要的。**）


```Python
a_gpu = cuda.mem_alloc(a.nbytes)
```


最后，咱们把刚刚生成的数组a转移到GPU里面吧：

```Python
cuda.memcpy_htod(a_gpu, a)
```


## 运行一个内核函数（kernel）


咱们这篇简介争取说的都是最简单的内容：咱们写一个代码来把a_gpu这段显存中存储的数组的每一个值都乘以2. 为了实现这个效果，我们就要写一段CUDA C代码，然后把这段代码提交给一个构造函数，这里用到了pycuda.compiler.SourceModule:




```Python
mod = SourceModule("""
 __global__ void doublify(float *a)
 {
 int idx = threadIdx.x + threadIdx.y*4;
 a[idx] *= 2;
 }
 """)
```

（**译者注：上面这段代码需要对C有一定了解。这里的threadIDx是CUDA C语言的内置变量，这里借此确定了a数组所在的位置，然后通过指针对数组中每一个元素进行了自乘2的操作。**）


这一步如果没有出错，就说明这段代码已经编译成功，并且加载到显卡中了。然后咱们可以使用一个到咱们这个pycuda.driver.Function的引用，然后调用此引用，把显存中的数组a_gpu作为参数传过去，同时设定块大小为4x4：

```Python
func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))
```





最后，咱们就把经过运算处理过的数据从GPU取回，并且将它和原始数组a一同显示出来对比一下：



```Python
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print (a_doubled)
print (a)
```

（**译者注：原作者在原文中使用的是Python2，我这里用的是带括号的print，这样同时能在python2和python3上运行。**）




输出的效果大概就是如下所示：

（**译者注：上面这个是从显存中取回的翻倍过的数组，下面的是原始数组。**）


```Bash
[[ 0.51360393  1.40589952  2.25009012  3.02563429]
 [-0.75841576 -1.18757617  2.72269917  3.12156057]
 [ 0.28826082 -2.92448163  1.21624792  2.86353827]
 [ 1.57651746  0.63500965  2.21570683 -0.44537592]]
[[ 0.25680196  0.70294976  1.12504506  1.51281714]
 [-0.37920788 -0.59378809  1.36134958  1.56078029]
 [ 0.14413041 -1.46224082  0.60812396  1.43176913]
 [ 0.78825873  0.31750482  1.10785341 -0.22268796]]
```



出现上面这样输出就说明成功了！整个攻略就完成了。另外很值得庆幸的是，运行输出之后PyCuda就会把所有清理和内存回收工作做好了，咱们的简介也就完毕了。不过你可以再看一下接下来的内容，里面有一些有意思的东西。


(本文的代码在PyCuda源代码目录下的examples/demo.py文件中。)



### 简化内存拷贝

PyCuda提供了pycuda.driver.In, pycuda.driver.Out, 以及pycuda.driver.InOut 这三个参数处理器（argument handlers），能用来简化内存和显存之间的数据拷贝。例如，咱们可以不去创建一个a_gpu，而是直接把a移动过去，下面的代码就可以实现：



```Python
func(cuda.InOut(a), block=(4, 4, 1))
```









### 有准备地调用函数


使用内置的 pycuda.driver.Function.__call__() 方法来进行的函数调用，会增加类型识别的资源开销（参考显卡接口）。 要实现跟上面代码同样的效果，又不造成这种开销，这个函数就需要设定好参数类型（如Python的标准库中的结构体模块struct所示），然后再去调用该函数。这样也就不用需要再使用numpy.number类去制定参数的规模了：


```Python
grid = (1, 1)
block = (4, 4, 1)
func.prepare("P")
func.prepared_call(grid, block, a_gpu)
```











## 抽象以降低复杂度

使用 pycuda.gpuarray.GPUArray，同样效果的代码实现起来就更加精简了：


```Python
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()
print a_doubled
print a_gpu
```









## 进阶内容


### 结构体

（由Nicholas Tung提供，代码在examples/demo_struct.py文件中)



假如我们用如下的构造函数，对长度可变的数组的每一个元素的值进行翻倍：



```Python
mod = SourceModule("""
 struct DoubleOperation {
 int datalen, __padding; // so 64-bit ptrs can be aligned
 float *ptr;
 };

 __global__ void double_array(DoubleOperation *a) {
 a = &a[blockIdx.x];
 for (int idx = threadIdx.x; idx datalen; idx += blockDim.x) {
 a->ptr[idx] *= 2;
 }
 }
 """)
```


网格grid中的每一个块block（这些概念参考CUDA的官方文档）都将对各个数组进行加倍。for循环允许比当前线程更多的数据成员被翻倍，当然，如果能够保证有足够多的线程的话，这样做的效率就低了。接下来，基于这个结构体进行封装出来的一个类就产生了，并且有两个数组被创建出来：



```Python
class DoubleOpStruct:
    mem_size = 8 + numpy.intp(0).nbytes
    def __init__(self, array, struct_arr_ptr):
        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        cuda.memcpy_htod(int(struct_arr_ptr), numpy.getbuffer(numpy.int32(array.size)))
        cuda.memcpy_htod(int(struct_arr_ptr) + 8, numpy.getbuffer(numpy.intp(int(self.data))))
    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

array1 = DoubleOpStruct(numpy.array([1, 2, 3], dtype=numpy.float32), struct_arr)
array2 = DoubleOpStruct(numpy.array([0, 4], dtype=numpy.float32), do2_ptr)
print("original arrays", array1, array2)
```





上面这段代码使用了pycuda.driver.to_device() 和 pycuda.driver.from_device() 这两个函数来分配内存和复制数值，并且演示了在显存中如何利用从已分配块位置进行的偏移。最后咱们执行一下这段代码；下面的代码中演示了两种情况：对两个数组都进行加倍，以及只加倍第二个数组：


```Python
func = mod.get_function("double_array")
func(struct_arr, block = (32, 1, 1), grid=(2, 1))
print("doubled arrays", array1, array2)

func(numpy.intp(do2_ptr), block = (32, 1, 1), grid=(1, 1))
print("doubled second only", array1, array2, "\n")
```


## 接下来的征程

当你对这些基础内容感到足够熟悉了，就可以去深入探索一下显卡接口。更多的例子可以再PyCuda的源码目录下的examples子目录。这个文件夹里面也包含了一些测试程序，可以用来比对GPU和CPU计算的差别。另外PyCuda源代码目录下的test子目录里面由一些关于功能如何实现的参考。




* [Github](https://github.com/inducer/pycuda)
* [Download Releases](https://pypi.python.org/pypi/pycuda)

©2008, Andreas Kloeckner.
©2016, translated to Chinese by [CycleUser](http://blog.cycleuser.org)
Powered by [Sphinx 1.4.8](http://sphinx-doc.org/) & [Alabaster 0.7.9](https://github.com/bitprophet/alabaster) | [Page source](https://documen.tician.de/pycuda/_sources/tutorial.txt#getting-started)



