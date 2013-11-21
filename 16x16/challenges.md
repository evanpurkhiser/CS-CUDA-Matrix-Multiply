## Challenges - CUDA blocked matrix multiplication

During the second half of this project where we were asked to convert the given
program to use 'blocked' matrix multiplication I faced a few challenges when
attempting to make things work. I've documented these here.

 1. I knew that I would need to add two products together in the global `d_P`
	array. Since I still had a very vague idea of how to get the 16x16 matrix
	working I wanted to ensure that swapping the `=` for a `+=` wasn't going to
	cause any problems. So I decided to try doing this on the first part of the
	project (using the 8x8 matrix).

	What I found was that this didn't work (all matrix indicies were incorrect).
	This was very perplexing as I was certian that the assignment (now
	addition assignment) was only being called once. I verified this by adding
	this line of code just before the reduction process:

	```c
      if(ty == 0) printf("assigning index %d. Current value is %f\n", tx * TILE_WIDTH + bx, d_P[tx * TILE_WIDTH + bx]);
	```

	Executing the program now I was able to see that each index in the product
	matrix was only being assigned once (as I suspected). **However**, I also
	saw that the `p_D` array was already filled with the answers _before
	actually doing the assignment_. I also noticed that the calculated numbers
	were two times what the correct answer was.

	After running the program again, I noticed that the calculated value was
	actually increasing in value. Instead of being twice that of the correct
	value this time it was three times.

	The problem here is that the memory on the video card is not implicitly
	cleared when you allocate it. Just the same way that main memory isn't
	implicity cleared when you allocate it with `malloc`. It's _undefined_.

	The solution is to add a call to `cudaMemset` in the `MatrixMultiplication`
	function to zero out the memory.

	```diff
	  // Allocate P on the device
	  cudaMalloc((void**) &Pd, size);
	+ cudaMemset(Pd, 0, size);
	```

 2. The second major challenge I ran into again involved the `+=` operation on
	the `d_P` array. At this point I was fairly sure that I had setup all of my
	tile position calculations correct and that the right values were being
	multiplied, however I was still getting incorrect answers for _all_ matrix
	indicies.

	During my time debuging this I had littered my program with various
	`printf` statments and at some point along the lines I noticed that I was
	actually **having less incorrect matrix indicies** as I added debug code.
	Finally, after adding a few more _completely unreleated debug lines_
	(outside of the device function even!) I was getting **no errors**.
	However, right after removing all of my debug code I was getting errors for
	the entire matrix again.

	Just as we had mentioned in class **there is a race condition when doing the
	addition assignment (`+=`) operation**. After some googling I discovered
	that CUDA has a _very nice_ `atomicAdd` function. Using this instead of the
	`+=` operator fixes the race condtion.

	```diff
   - if (ty == 0) d_P[(tx + x_offset) * Width + bx + bx_offset] += partialSum[tx][ty] + partialSum[tx][ty + 1];
   + if (ty == 0) atomicAdd(&d_P[(tx + x_offset) * Width + bx + bx_offset], partialSum[tx][ty] + partialSum[tx][ty + 1]);
	```
