dnsmb1: dnsmb1.cu
	/opt/cuda/bin/nvcc -I /opt/cuda/samples/common/inc/ dnsmb1.cu -o dnsmb1 -arch=sm_20
