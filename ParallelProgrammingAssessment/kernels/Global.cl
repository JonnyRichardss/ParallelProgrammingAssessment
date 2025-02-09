kernel void createHistogram_Global(global DATA_TYPE* A, global HIST_TYPE* GlobalHistogram) {
	int gid = get_global_id(0);

	if (gid < IMAGE_SIZE) {
		HIST_TYPE bin = (A[gid] * NUM_BINS) / (1 << BIT_DEPTH); //if the number of bins is not equivalent to the bit depth's value range, rescale the value
		atom_inc(&GlobalHistogram[bin]);
	}
}
//basic version of cumulation based on scan_hs from tutorial 3
//Please note - this only works when workgroup_size == num_bins
kernel void AccumulateHistogram_Global(global HIST_TYPE* A, global HIST_TYPE* B) {
	int gid = get_global_id(0);
	if (gid < NUM_BINS) {
		global HIST_TYPE* C;
		for (int i = 1; i < NUM_BINS; i *= 2) {
			B[gid] = A[gid];
			if (gid >= i)
				B[gid] += A[gid - i];
			barrier(CLK_GLOBAL_MEM_FENCE);
			C = A;
			A = B;
			B = C;
		}
	}
}
kernel void AccumulateHistogram_SingleGroup(global HIST_TYPE* A, global HIST_TYPE* B) {
	//entirely the same principle as tutorial version and above version
	//we have to force to a single thread group since different groups cannot be synchronised
	int wid = get_group_id(0);
	int lid = get_local_id(0);
	if (wid == 0) {
		global HIST_TYPE* C;
		for (int i = 1; i < NUM_BINS; i *= 2) {
			for (int j = lid; j < NUM_BINS; j += get_local_size(0))//stride to account for local group size being different to work size
			{
				B[j] = A[j];
				if (j >= i)
					B[j] += A[j - i];
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
			C = A;
			A = B;
			B = C;
		}
	}
}

kernel void NormalizeHistogram_Global(global HIST_TYPE* A) {
	int gid = get_global_id(0);
	if (gid < NUM_BINS) {
		HIST_TYPE max_val = A[NUM_BINS - 1];
		A[gid] = (A[gid] * (1 << BIT_DEPTH)) / max_val;
	}
	
}
kernel void ApplyHistogram_Global(global DATA_TYPE* A, global HIST_TYPE* Hist) {
	int gid = get_global_id(0);
	if (gid < IMAGE_SIZE) {
		HIST_TYPE bin = (A[gid] * NUM_BINS) / (1 << BIT_DEPTH); //if the number of bins is not equivalent to the bit depth's value range, rescale the value
		const HIST_TYPE MaxVal = (1 << BIT_DEPTH) - 1;//clamp to prevent overflow
		A[gid] = min(Hist[bin], MaxVal);
	}

}