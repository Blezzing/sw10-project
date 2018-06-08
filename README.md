# YAGAL
Repository containing the implementation work of our thesis.

# Repository Structure
The repository is seperated into 3 directories.

## code
The code directory contains the implementation of yagal with a  sample program using it. Inside this directory there is a subdirectory named `yagal`which contains the actual implementation of the framework.

Reading the makefile in the code directory show an example of the process necessary for the compilation.

## saxpy
The saxpy directory contains the example implementations shown in the thesis, including makefiles for compilation.

## scribbles
The scribbles directory contain small files with ideas and code samples for how we initially wanted yagal to be used.

# Requirements For Compilation
To compile a program using yagal you need the following installed:

* CUDA Toolkit (To get the CUDA Driver API) from developer.nvidia.com/cuda-toolkit
* LLVM 7.0.0svn (To get the LLVM libraries) from
github.com/llvm-mirror/llvm

# Compilers
We have only used Clang and GCC to compile yagal programs, but other should work as well.