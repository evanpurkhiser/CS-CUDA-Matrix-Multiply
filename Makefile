CUDA=/opt/cuda

dnsmb1: dnsmb1.cu product.txt matrix1.txt matrix2.txt
	$(CUDA)/bin/nvcc -I $(CUDA)/samples/common/inc/ dnsmb1.cu -o dnsmb1 -arch=sm_20

%.txt:
	wget http://www.cs.uakron.edu/~toneil/cs477/Labs/lab5/$@

report.pdf: report.md
	pandoc report.md -V geometry:margin=1in -o report.pdf

clean:
	rm -f dnsmb1 *.txt report.pdf
