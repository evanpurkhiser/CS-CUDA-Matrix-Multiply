dnsmb1: dnsmb1.cu product.txt matrix1.txt matrix2.txt
	/opt/cuda/bin/nvcc -I /opt/cuda/samples/common/inc/ dnsmb1.cu -o dnsmb1 -arch=sm_20

%.txt:
	wget http://www.cs.uakron.edu/~toneil/cs477/Labs/lab5/$@

clean:
	rm  dnsmb1 *.txt
