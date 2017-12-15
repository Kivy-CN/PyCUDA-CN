Title: MacOS PyCUDA Python Pyenv
Date: 2016-10-21 11:20
Category: Python
Tags: Python,CUDA,Mac,Pyenv

####Mac系统下使用Pyenv管理Python多版本，并且给各个版本安装PyCUDA


本文是针对PyCUDA的新手用户。此处特点是使用了Pyenv构建了多个工作环境，并且指导如何在各个不同的Python环境中安装PyCUDA。

####安装Git和Pyenv

下载PyCUDA代码需要用Git，管理多版本的Python需要Pyenv，而这两个的安装就都需要用[Brew](http://brew.sh/index_zh-cn.html) 了。在终端输入下面的命令就可以安装Brew了:

```Bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

然后再接着在终端陆续输入下面两个命令来安装Git和Pyenv：

```Bash
brew install git
brew install pyenv
```
####最最重要的一步

这一步是最重要的了，决定了你能否成功安装和运行CUDA以及PyCUDA。要运行Brew，你就被迫要安装最新版本的Xcode和配套的Command Line Tools，但是CUDA很可能和这个最新版本不兼容。所以如果你有旧版本的Xcode，一定要备份一下，改个名字别被替换了啥的。然后安装最新的Xcode和配套的Command Line Tools之后，赶紧用Brew安装好Git和Pyenv。安装好了这两个之后，就降级回到能兼容CUDA的旧版本Xcode，重新下载安装旧版本的Command Line Tools。并且绝对别在App Store里面把它升级到最新版。

只有安装好了Pyenv和能够支持运行CUDA的旧版本Xcode以及Command Line Tools ，我们才能完成CUDA的安装.

####安装Xcode, Command Line Tools以及CUDA

我之前的[文章](http://blog.cycleuser.org/use-cuda-80-with-macos-sierra-1012.html)中更详细地讲解了关于CUDA和Xcode的兼容情况以及解决方案，我正打字这回，情况依然还是跟这篇文章中一样。最先帮你把的CUDA依然不能使用Xcode8，需要安装Xcode7.3.1 和 Command Line Tools for XCode 7.3.1，可以在 [这里](https://developer.apple.com/download) 找到官方提供的下载链接。

一定要确保你安装的Xcode是能够支持CUDA运行的。然后才能成功安装 [CUDA](https://developer.nvidia.com/compute/cuda/8.0/Prod/local_installers/cuda_8.0.47_mac-dmg)，安装CUDA之后运行下面这个命令来检查一下环境变量是否设置正确:
```Bash
nvcc --version
```
没问题的话应该显示类似下面的结果：
```Bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sun_Sep_18_22:16:08_CDT_2016
Cuda compilation tools, release 8.0, V8.0.46
```


####使用Pyenv安装一个Python副本

在终端中输入下面的命令查看可以用Pyenv安装的全部Python版本：
```Bash
pyenv install --list
```

这里我用3.5.2做一个例子：
```Bash
pyenv install 3.5.2
```
输入上述命令，等待完成之后，我们就有了一个全新的Python环境了，怎么折腾都可以，不会影响系统的Python配置。

####安装PyCUDA

把下面的命令粘贴到终端中来下载PyCUDA的源代码：
```Bash
git clone --recursive http://git.tiker.net/trees/pycuda.git
```
进入到pycuda的目录并且设置目录内的Python为刚刚咱们安装的3.5.2版本：

```Bash
cd pycuda
pyenv local 3.5.2
```

接下来用下列命令来配置、编译、安装：
```Bash
python configure.py
sudo make
sudo make install
```

如果没有发现报错，就应该是成功了。把下面的代码保存到一个名为test.py的文件中，然后咱们来测试一下：
```Python
# Sample source code from the Tutorial Introduction in the documentation.
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
a_gpu = gpuarray.to_gpu(numpy.random.randn(4, 4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()

print("original array:")
print(a_gpu)
print("doubled with gpuarray:")
print(a_doubled)
```


在终端中运行这个test.py，如果得到类似下面这样的结果，就是成功了：
```Bash
$ python test.py
original array:
[[ 0.27740544 -1.44831014  0.6379782   0.15358959]
 [-0.21130283 -0.19202329 -2.23594046  0.14036565]
 [-0.69078982 -0.44290611  1.2644769   1.55474603]
 [-1.08704031  2.22870898  0.85237521  0.15609477]]
doubled with gpuarray:
[[ 0.55481088 -2.89662027  1.27595639  0.30717918]
 [-0.42260566 -0.38404658 -4.47188091  0.28073129]
 [-1.38157964 -0.88581222  2.52895379  3.10949206]
 [-2.17408061  4.45741796  1.70475042  0.31218955]]
```


####更多版本

如果你要安装PyCUDA到更多版本的Python中，只要用Pyenv来安装更多版本的Python，然后把pycuda所在目录设置为对应版本的Python，之后重复上面的配置、编译、安装的步骤就可以了。例如下面就用3.5.1做例子示范了一下：
```Bash
pyenv install 3.5.1
cd ~/pycuda
pyenv local 3.5.1
python configure.py
sudo make
sudo make install
```
就是这样了。


