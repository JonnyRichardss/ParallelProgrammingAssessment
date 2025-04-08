#pragma once
#include "ImageProcessorKernel.h"
#include <chrono>
//These exist to allow me to feed in the desired image data type to the CL compiler
template<typename T>
std::string GetCLTypename()
{
	return "";
}
template<>
std::string GetCLTypename<unsigned char>()
{
	return "uchar";
}
template<>
std::string GetCLTypename<unsigned short>()
{
	return "ushort";
}template<>
std::string GetCLTypename<unsigned int>()
{
	return "uint";
}

template <typename CIMG_TYPE>
class ImageProcessor
{
public:
	//Constructors, Destructors
	ImageProcessor(int platform_id, int device_id, int workgroup_size,int _num_bins, std::string& image_filename, std::string& kernel_folder, bool useProfiling, bool _ignoreColour,bool _displayHistograms)
		:group_size(workgroup_size),
		profilingEnabled(useProfiling),
		num_bins(_num_bins),
		ignoreColour(_ignoreColour),
		displayHistograms(_displayHistograms),
		inputPath(image_filename)
	{
		//load images
		inputImage = CImg::CImg<CIMG_TYPE>(image_filename.c_str());
		outputImage = CImg::CImg<CIMG_TYPE>(inputImage.width(), inputImage.height(), inputImage.depth(), inputImage.spectrum());
		//setup openCL program
		context = Utils::GetContext(platform_id, device_id);
		Utils::AddAllSources(sources, kernel_folder);
		queue = cl::CommandQueue(context, useProfiling ? CL_QUEUE_PROFILING_ENABLE : 0U);
		program = cl::Program(context, sources);

		//TODO we know the input size before we compile -- we can pass the input size in as a #define to the compiler along with other needed info like num buckets 
		//build openCL program
		try {
			std::stringstream compileOptions;
			compileOptions << "-D NUM_BINS=" << num_bins << " ";
			compileOptions << "-D BIT_DEPTH=" <<  sizeof(CIMG_TYPE)* 8 << " ";
			compileOptions << "-D DATA_TYPE=" << GetCLTypename<CIMG_TYPE>() << " ";
			compileOptions << "-D HIST_TYPE=" << STR(HIST_TYPE) << " ";
			compileOptions << "-D IMAGE_SIZE=" << inputImage.size() << " ";
			program.build(compileOptions.str().c_str());
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//setup openCL I/O
		ImageBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, inputImage.size() * sizeof(CIMG_TYPE));
		//allocated device buffers for histograms
		histogramA = cl::Buffer(context, CL_MEM_READ_WRITE, num_bins * sizeof(HIST_TYPE));
		histogramB = cl::Buffer(context, CL_MEM_READ_WRITE, num_bins * sizeof(HIST_TYPE));
	}
	virtual ~ImageProcessor() {}
public:
	//Publicly accessible functions
	void AddKernel(ImageProcessorKernel<CIMG_TYPE>* kernel) {
		kernel->Init(program, inputImage, outputImage, queue, ImageBuffer, histogramA, histogramB, num_bins,group_size,ignoreColour,displayHistograms);
		allKernels.push_back(kernel);
	}
	void RunAll() {
		if (allKernels.size() == 0) {
			std::cerr << "No kernels added to image processor!" << std::endl;
			return;
		}
		std::chrono::time_point start = std::chrono::high_resolution_clock::now();
		for (ImageProcessorKernel<CIMG_TYPE>* kernel : allKernels) {
			std::cout << "\nNow Running kernel: " << kernel->GetName() << "!" << std::endl;
			kernel->Run(profilingEnabled);
		}
		std::chrono::time_point end = std::chrono::high_resolution_clock::now();
		if (!profilingEnabled) return;
		std::cout << "\nTotal execution time for all kernels [ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;

	}
	void DisplayImages() {
		CImg::CImgDisplay disp_input(inputImage, "Input");
		CImg::CImgDisplay disp_output(outputImage, "Output");
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC())
		{
			disp_input.wait(1);
			disp_output.wait(1);
		}
		outputImage.save(("equalized_" + inputPath).c_str());
	}
protected:
	std::string inputPath;
	bool profilingEnabled;
	bool displayHistograms;
	bool ignoreColour;
	int group_size;
	int num_bins;
	cl::Program::Sources sources;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	
	cl::Buffer ImageBuffer;

	cl::Buffer histogramA;
	cl::Buffer histogramB;

	std::vector<ImageProcessorKernel<CIMG_TYPE>*> allKernels;

	CImg::CImg<CIMG_TYPE> inputImage;
	CImg::CImg<CIMG_TYPE> outputImage;

public:
	//(filling out rule of 5)
	ImageProcessor(const ImageProcessor& other) = delete;//copy constructor
	ImageProcessor(ImageProcessor&& other) noexcept = delete;//move constructor
	ImageProcessor& operator=(const ImageProcessor& other) = delete;//copy assignment operator
	ImageProcessor& operator=(ImageProcessor&& other) noexcept = delete;//move assignment operator
};



