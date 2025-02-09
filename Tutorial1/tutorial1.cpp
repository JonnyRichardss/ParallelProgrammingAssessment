//g++ -std=c++0x tutorial1.cpp -o tutorial1 -lOpenCL
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl.hpp>
#include "Utils.h"

#include <iostream>
#include <vector>

void print_help() {
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	
	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);


		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		//catch (const cl::Error& err) {
		catch (...) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			//throw err;
		}
		constexpr int NUM_EXECS = 5;
		constexpr int NUM_COPIES = NUM_EXECS * 3;
		constexpr int ARR_SIZE = 500000000;
		std::cout << "Running " << NUM_EXECS << " times with " << ARR_SIZE << " elements.\n\n" << std::endl;
		std::vector<cl_ulong> execTimes;
		std::vector<cl_ulong> copyTimes;
		execTimes.reserve(NUM_EXECS);
		copyTimes.reserve(NUM_COPIES);
		for (int i = 0; i < NUM_EXECS; i++) {
			//Part 3 - memory allocation
			//host - input
			 
			//Small size vectors with custom values
			//std::vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //C++11 allows this type of initialisation
			//std::vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
			//Large size vectors zero initialized
			std::vector<int> A(ARR_SIZE);
			std::vector<int> B(ARR_SIZE);

			size_t vector_elements = A.size();//number of elements
			size_t vector_size = A.size() * sizeof(int);//size in bytes

			//host - output
			std::vector<int> C(vector_elements);

			//device - buffers
			cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
			cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
			cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

			//Part 4 - device operations
			cl::Event AEvent;
			cl::Event BEvent;
			cl::Event CEvent;
			cl::Event ExecEvent;

			//4.1 Copy arrays A and B to device memory
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0],nullptr, &AEvent);
			copyTimes.push_back(AEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - AEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0],nullptr, &BEvent);
			copyTimes.push_back(BEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - BEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			//4.2 Setup and execute the kernel (i.e. device code)
			cl::Kernel kernel_add = cl::Kernel(program, "add");
			kernel_add.setArg(0, buffer_A);
			kernel_add.setArg(1, buffer_B);
			kernel_add.setArg(2, buffer_C);

			queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, nullptr, &ExecEvent);
			//4.3 Copy the result from device to host
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0],nullptr, &CEvent);
			copyTimes.push_back(CEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - CEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			execTimes.push_back(ExecEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ExecEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
		}
		cl_ulong totalExecTime = 0;
		cl_ulong totalCopyTime = 0;
		for (cl_ulong time : execTimes) {
			totalExecTime += time;
		}for (cl_ulong time : copyTimes) {
			totalCopyTime += time;
		}
		totalExecTime /= NUM_EXECS;
		totalCopyTime /= NUM_COPIES;
		std::cout <<  "Average time over " << NUM_EXECS << " runs [ns]: " << totalExecTime << std::endl;
		std::cout <<  "Average time over " << NUM_COPIES << " copies [ns]: " << totalCopyTime << std::endl;
		std::cout <<  "Total average [ns]: " << totalExecTime + totalCopyTime << std::endl;
	}
	//catch (cl::Error err) {
	catch (...) {
		//std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
