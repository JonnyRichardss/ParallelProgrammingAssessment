#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -w : set workgroup size (default 64)" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	int workgroup_size = 64;
	string image_filename = "test.ppm";
	string kernel_function = "filter_r_counter_glob";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-w") == 0) && (i < (argc - 1))) { workgroup_size = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-k") == 0) && (i < (argc - 1))) { kernel_function = argv[++i]; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		std::cout << "Image size:\nwidth: " << image_input.width() << "\nheight: " << image_input.height() << "\ndepth: " << image_input.depth() << "\ncols: " << image_input.spectrum() << std::endl;
		CImgDisplay disp_input(image_input,"input");

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;


		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer dev_int_output(context, CL_MEM_READ_WRITE, sizeof(int)); //should be the same as input image
		cl::Buffer dev_size_input(context, CL_MEM_READ_ONLY, sizeof(int)); //should be the same as input image
//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));
		int size = image_input.size();
		int zero = 0;
		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(dev_size_input, CL_TRUE, 0, sizeof(int), &size);
		queue.enqueueWriteBuffer(dev_int_output, CL_TRUE, 0, sizeof(int), &zero);
//		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, kernel_function.c_str());
		cout << "Max Group size: " << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
		std::cout << "Current Group size: " << workgroup_size << endl;
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);
		kernel.setArg(2, dev_int_output);
		kernel.setArg(3, cl::Local(sizeof(int)));
		kernel.setArg(4, dev_size_input);
//		kernel.setArg(2, dev_convolution_mask);
		cl::Event kernelEvent;
		int global_size = size;
		if (global_size % workgroup_size != 0) {
			global_size += workgroup_size - (global_size % workgroup_size);
		}
		std::cout << "Image Size was : " << size << " Global Range set to: " << global_size << endl;
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(workgroup_size),nullptr,&kernelEvent);

		//vector<unsigned char> output_buffer(image_input.size());
		//CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//4.3 Copy the result from device to host
		CImg<unsigned char> output_image(image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		int totalChannels = 0;
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_image.size(), &output_image.data()[0]);
		queue.enqueueReadBuffer(dev_int_output, CL_TRUE, 0, sizeof(int), &totalChannels);
		CImgDisplay disp_output(output_image,"output");
		std::cout << "Exec time (ns): " << kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Number of pixels: " << totalChannels << "(Expected " << image_input.width() * image_input.height() << ")" << std::endl;
 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
