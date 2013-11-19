# CUDA Matrix Multiplication

Evan Purkhiser

In this lab I will be calculating the product of two matrixes using the NVIDIA
CUDA toolkit. I will be using the 8 block 8x8 thread variation of the matrix
multiplication program for my trials.

I will be performing a total of four experiments. Half of the experiments will
be performed on the University of Akron 'Tesla' server, which uses a NVIDIA
C2050 Tesla card. The other half of the experiments will be performed on one of
the Computer Science lab machines which utilize a NVIDIA GTX 480 consumer
graphics card. As a bonus I will also do some small comparisons to my desktop
machine that utilizes a NVIDIA GTX 760 card. On these machines I will be
recording the time trials of running the program without blocked matrix
multiplication (as is), and then using blocked matrix multiplication.

## Linux

Tun run the program on the Tesla server as well as on my Linux desktop a small
change had to be made to the program itself. The Makefile to build the program
also needed to be changed slightly to account for variations in the installation
path of the CUDA toolkit.

First, since we are using CUDA 5 we must changed the `cutil.h` include to
`helper_cuda.h`.

To handle compiling the program on Linux I created a small Makefile that will 

```makefile
CUDAPATH=/opt/cuda

dnsmb1: dnsmb1.cu product.txt matrix1.txt matrix2.txt
	$(CUDAPATH)/bin/nvcc -I $(CUDAPATH)/samples/common/inc/ dnsmb1.cu -o dnsmb1 -arch=sm_20

%.txt:
	wget http://www.cs.uakron.edu/~toneil/cs477/Labs/lab5/$@
```

Note that I've included a `CUDAPATH` variable that indicates the location of the
CUDA toolkit installation directory. On my machines CUDA is installed into
`/opt/cuda` while on the tesla server it is installed to `/usr/local/cuda`. This
allows us to easily change the CUDA path for build time.

I've also added a build target to download the required testing data files
(matrix{1,2}.txt and product.txt).

## Experiment 1

For the first experiment I ran the program and recorded execution times without
making any changes.

I ran the program with the command `for r in {1..10}; do ./dnsmb1; done`

| Trial 1      | Trial 2      | Trial 3      | Trial 4      | Trial 5      | Trial 6      | Trial 7      | Trial 8      | Trial 9      | Trial 10     |
| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------|
| $285.952\mu$ | $281.440\mu$ | $281.440\mu$ | $284.832\mu$ | $284.288\mu$ | $279.296\mu$ | $285.120\mu$ | $287.008\mu$ | $285.664\mu$ | $286.656\mu$ |

| Minimum time | Maximum time | Average Time | STD      |
| ------------ | ------------ | ------------ | -------- |
| $279.269\mu$ | $287.008\mu$ | $284.200\mu$ | $2.575$  |

And just for fun here is some bonus data from this same program being run on my
desktop Linux machine.

| Minimum time | Maximum time | Average Time | STD        |
| ------------ | ------------ | ------------ | ---------- |
| $114.112\mu$ | $495.808\mu$ | $189.300\mu$ | $136.4\mu$ |

### Discussion

The primary thing we can take note of here is just how consistent the times were
on the Tesla server. With only a standard deviation of $2.5\mu s$ we can see
that the CUDA program doesn't have a lot of play in execution time.

Just for fun I decided to let the tesla server do some more work and executed
the command `time while true; do ./dnsmb1; done > times.txt` and let it run for
a little while. The results from this showed again a standard deviation of just
$3.292$. That's out of a total of 123 trials. (The `time` failed to report to
standard error for some reason however).

Comparing this to my desktop Linux machine we can see there was actually a much
higher standard deviation of the 10 trials, with the maximum time being more
than 4 times that of the minim execution time. I sepculate that this could be in
part due to the difference in the "consumer" grade graphics cards versus the
"server" grade devices.

Something else to note that wasn't shown in the results however was that the
startup time for the program was actually much faster on my desktop machine than
it was on the tesla server. On my desktop executing `time ./dnsmb1` gives me a
real time of $0.050s$, while on the tesla server it actually takes $2.277s$. I
did a quick investigation into this by using the `strace` utility with the -tt
option to display the time stamp (with microseconds) for each system call. I was
able to clearly see that the tesla server hung on the `open("/dev/nvidia0",
O_RDWR)` call, returning a `EINTR` indicating that the device was actually busy.

