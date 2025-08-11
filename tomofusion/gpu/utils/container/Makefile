#Makefile
#!/bin/sh

include ./make.inc

all: matrix_op matrix3d 

matrix_op: matrix_ops.cu matrix_ops.h
	$(CUDAXX) matrix_ops.cu 

matrix3d: Matrix3D.o Matrix3D.h 
	$(CXX) $(CXXFLAGS)  Matrix3D.cpp matrix_ops.o 

clean:
	rm -rf *.so *.o *.out
