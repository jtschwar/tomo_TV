#Makefile
#!/bin/sh

include ./make.inc

all: matrix_op matrix4d

matrix_op: matrix_ops.cu matrix_ops.h
	$(CUDAXX) matrix_ops.cu 

matrix4d: Matrix4D.o Matrix4D.h
	$(CXX) $(CXXFLAGS) Matrix4D.cpp matrix_ops.o

clean:
	rm -rf *.so *.o
