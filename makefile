#!/bin/bash
all:	mat_mult.cu
	nvcc -o exe mat_mult.cu
clean:
	$(RM) exe
