//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	if (colour_channel == 0) B[id] = A[id];
	else return;
}
kernel void filter_r_counter_loc(global const uchar* A, global uchar* B, global int* totalChannels,local int* localChannels,global int* inputSize) {
	int id = get_global_id(0);
	int image_size = *inputSize/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	if (get_global_id(0) == 0) {
		printf("\n%i\n", *inputSize);
	}
	if (get_local_id(0) == 0) {
		*localChannels = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	//this is just a copy operation, modify to filter out the individual colour channels
	if (get_global_id(0) < *inputSize) {
		if (colour_channel == 0) {
			B[id] = A[id];
			atomic_inc(localChannels);
		}
	}
	else {
		//printf("I");
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0) {
		atomic_add(totalChannels, *localChannels);
	}
}
kernel void filter_r_counter_glob(global const uchar* A, global uchar* B, global int* totalChannels, local int* localChannels, global int* inputSize) {
	int id = get_global_id(0);
	int image_size =*inputSize/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	if (get_global_id(0) == 0) {
		printf("\n%i\n", *inputSize);
	}
	//this is just a copy operation, modify to filter out the individual colour channels
	if (id < *inputSize) {
		if (colour_channel == 0) {
			B[id] = A[id];
			atomic_inc(totalChannels);
		}
	}
	else {
		//printf("I");
	}
}
kernel void invert(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = 255 - A[id];
}
kernel void greyscale(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	int rID = id - (colour_channel * image_size);
	B[id] = (0.2126 * A[rID]) + (0.7152 * A[rID + image_size]) + (0.0722 * A[rID + (image_size * 2)]);
	//printf("Col, %i, Val: %u\n", colour_channel,Oput);
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {

	int3 size = (int3)(get_global_size(0),get_global_size(1),get_global_size(2));
	int3 pos = (int3)(get_global_id(0),get_global_id(1),get_global_id(2));

	int z = pos.z * size.x * size.y;
	int id = pos.x + pos.y*size.x + z; //global id in 1D space
	
	uint result = 0;

	for (int x = (pos.x-1); x <= (pos.x+1); x++)
	for (int y = (pos.y-1); y <= (pos.y+1); y++) 
		result += A[x + y*size.x + z];

	result /= 9;

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];

	B[id] = (uchar)result;
}