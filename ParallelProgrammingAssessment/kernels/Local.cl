kernel void createHistogram(global DATA_TYPE* A, global HIST_TYPE* GlobalHistogram, local HIST_TYPE* LocalHistogram) {
	int lid = get_local_id(0);
	int gid = get_global_id(0);
	if (gid < IMAGE_SIZE) {
		//if there are less threads than bins, we need to use a stride to get them all
		for (int i = lid; i < NUM_BINS; i += get_local_size(0))
		{
			//clear local histogram
			LocalHistogram[i] = 0;
		}
		barrier(CLK_LOCAL_MEM_FENCE); //sync so that whole local histogram is cleared

		//atomically create local histogram

		HIST_TYPE bin = (A[gid] * NUM_BINS) / (1 << BIT_DEPTH); //if the number of bins is not equivalent to the bit depth's value range, rescale the value
		atom_inc(&LocalHistogram[bin]);

		barrier(CLK_LOCAL_MEM_FENCE); //sync for local histogram to complete

		for (int i = lid; i < NUM_BINS; i += get_local_size(0))
		{
			//atomically add local histogram to global
			atom_add(&GlobalHistogram[i], LocalHistogram[i]);
		}
		//no need to sync if we return to host here
		//barrier(CLK_GLOBAL_MEM_FENCE); //sync after creating global histogram
	}
}

//accumulation is done through scanning - the intermediate scan of block sums is implied in the second step
//(instead of explicitly scanning block sums, the input is just used directly later on)
kernel void AccumulateHistogram_1(global HIST_TYPE* A, global HIST_TYPE* B, local HIST_TYPE* localA,local HIST_TYPE* localB) {
	// 1 - accumulate block (size determined by workgroup size)
	int lid = get_local_id(0);
	int gid = get_global_id(0);
	int block = get_group_id(0);
	if (gid < NUM_BINS) {


		localA[lid] = A[gid]; //initial copy
		local HIST_TYPE* localC;
		for (int i = 1; i < get_local_size(0); i *= 2) {

			localB[lid] = localA[lid];
			if (lid >= i)
				localB[lid] += localA[lid - i];
			barrier(CLK_LOCAL_MEM_FENCE);
			localC = localA;
			localA = localB;
			localB = localC;

		}

		barrier(CLK_GLOBAL_MEM_FENCE);
		//localA is now the output for current block
		B[gid] = localA[lid];
	}
}
kernel void AccumulateHistogram_2(global HIST_TYPE* A, global HIST_TYPE* B) {
	//read from input add to output (map / maybe gather pattern) - no races
	//add the max of each previous block to each value in output
	int gid = get_global_id(0);
	int block = get_group_id(0);
	int localSize = get_local_size(0);
	int maxLid = localSize - 1;
	if (gid < NUM_BINS) {
		//copy input to output
		B[gid] = A[gid];
		//add all previous block sums to output
		for (int i = 0; i < block; i++) {
			B[gid] += A[(i * localSize) + maxLid];
		}
	}
}
//This is entirely identical to NormalizeHistogram_Global
//There is no need for local memory to be used since each output bin is set once (map pattern) - no race condition can occur
kernel void NormalizeHistogram(global HIST_TYPE* A) {
	int gid = get_global_id(0);
	if (gid < NUM_BINS){
		HIST_TYPE max_val = A[NUM_BINS - 1];
		A[gid] = (A[gid] * (1 << BIT_DEPTH)) / max_val;
	}
}

//This is entirely identical to ApplyHistogram_Global
//There is no need for local memory to be used since each output pixel is set once (map pattern) - no race condition can occur
kernel void ApplyHistogram(global DATA_TYPE* A, global HIST_TYPE* Hist) {
	int gid = get_global_id(0);
	if (gid < IMAGE_SIZE) {
		HIST_TYPE bin = (A[gid] * NUM_BINS) / (1 << BIT_DEPTH); //if the number of bins is not equivalent to the bit depth's value range, rescale the value
		const HIST_TYPE MaxVal = (1 << BIT_DEPTH) - 1;//clamp to prevent overflow
		A[gid] = min(Hist[bin], MaxVal);
	}

}