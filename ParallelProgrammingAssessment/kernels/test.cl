
#define MY_TYPE uchar

kernel void testKernel(global const MY_TYPE* A, global MY_TYPE* B) {
	int id = get_global_id(0);
	B[id] = 255 - A[id];
	//printf("A: %u B: %u\n", A[id], B[id]);
}