CUDA=/opt/cuda

dnsmb1: dnsmb1.cu product.txt matrix1.txt matrix2.txt
	$(CUDA)/bin/nvcc -I $(CUDA)/samples/common/inc/ dnsmb1.cu -o dnsmb1 -arch=sm_20

%.txt:
	wget http://www.cs.uakron.edu/~toneil/cs477/Labs/lab5/$@

clean:
	rm -r dnsmb1 *.txt
