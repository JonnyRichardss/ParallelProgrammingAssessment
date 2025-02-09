#include <iostream>
#include "ImageProcessor.h"
int main(int argc, char** argv)
{
	//process arguments
	int platform_id = 0;
	int device_id = 0;
	int workgroup_size = 256;
	int num_bins = 256;
	bool highDepth = false;
	bool ignoreColour = false;
	std::string image_filename = "test.pgm";
	std::string kernel_folder = "kernels";
	bool profilingEnabled = true;
	bool showGraphs = false;
	for (int i = 1; i < argc; i++) {
		if      ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-w") == 0) && (i < (argc - 1))) { workgroup_size = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { num_bins = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-i") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { kernel_folder = argv[++i]; }
		else if ((strcmp(argv[i], "-t") == 0					)) { profilingEnabled = false; }
		else if ((strcmp(argv[i], "-h") == 0					)) { highDepth = true; }
		else if ((strcmp(argv[i], "-g") == 0					)) { showGraphs = true; }
		else if ((strcmp(argv[i], "-c") == 0					)) { ignoreColour = true; }
		else if ((strcmp(argv[i], "-l") == 0					)) { std::cout << Utils::ListPlatformsDevices() << std::endl; return 0; }
		else if ((strcmp(argv[i], "-h") == 0                    )) { Utils::print_help(); return 0; }
		else													   { std::cout << "Unknown option: " << argv[i] << std::endl; return 0; }
	}
	std::cout
		<< "Running on " << Utils::GetPlatformName(platform_id) << ", " << Utils::GetDeviceName(platform_id, device_id) << "\n"
		<< "Workgroup size: " << workgroup_size << "  Number of Bins: " << num_bins << "\n"
		<< "Colour channels " << (ignoreColour ? "ignored" : "calculated separately") << "\n"
		<< "Image: " << image_filename << "    Processed as " << (highDepth ? "high bit depth (16)" : "low bit depth (8)") << "\n"
		<< "Profiling " << (profilingEnabled ? "enabled" : "disabled") << "  Graphs " << (showGraphs ? "shown" : "hidden") << "\n"
		<< std::endl;

	//Run main program 
	try {
		if (highDepth) {
			ImageProcessor<unsigned short> processor(platform_id, device_id, workgroup_size, num_bins, image_filename, kernel_folder, profilingEnabled, ignoreColour,showGraphs);
			GlobalKernel  <unsigned short> G;
			LocalKernel   <unsigned short> L;
			processor.AddKernel(&G);
			processor.AddKernel(&L);
			processor.RunAll();
			processor.DisplayImages();
		}
		else {
			ImageProcessor<unsigned char> processor(platform_id, device_id, workgroup_size, num_bins, image_filename, kernel_folder, profilingEnabled,ignoreColour,showGraphs);
			GlobalKernel  <unsigned char> G;
			LocalKernel   <unsigned char> L;
			processor.AddKernel(&G);
			processor.AddKernel(&L);
			processor.RunAll();
			processor.DisplayImages();
		}
	}
	//Display error and exit on all thrown OpenCL and CImage exceptions
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << Utils::getErrorString(err.err()) << std::endl;
		exit(err.err());
	}
	catch (CImg::CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
		exit(1);
	}
	return 0;
}

