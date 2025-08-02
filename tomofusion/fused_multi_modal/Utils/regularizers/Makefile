#Makefile
#!/bin/sh

include ./make.inc

all: tv_gd tv_fgp 

tv_gd: tv_gd.cu tv_gd.h
	$(CUDAXX) tv_gd.cu 

tv_fgp: tv_fgp.cu tv_fgp.h 
	$(CUDAXX) tv_fgp.cu  

tv_sb: tv_sb.cu tv_sb.h
	$(CUDAXX) tv_sb.cu

clean:
	rm -rf *.so *.o
