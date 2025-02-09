#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include "Utils.h"
#include "Vendor/CImg.h"
//this class only exists so that I can run many different versions of the algorithm from a single version of the ImageProcessor class
//its slightly over-engineered but it's not that deep that I need to find a "perfect" way to make it all go

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong; //i developed this in VS22 and in windows a single long is still 32 bits ty microsoft
//#define CIMG_TYPE uchar
//#define CIMG_TYPE ushort
//#define HIST_TYPE uint
#define HIST_TYPE ulong
//https://stackoverflow.com/questions/47346133/how-to-use-a-define-inside-a-format-string
//for passing above into CL compiler
#define STR_(X) #X
#define STR(X) STR_(X)
//rename namespace to something easier instead of just a "using namespace"
namespace CImg = cimg_library;

//because of the templating I have to stack the class and the impl into the same header


template<typename CIMG_TYPE>
class ImageProcessorKernel {
public:
	//Constructors, Destructors
	ImageProcessorKernel(const char* _kernelName) : kernelName(_kernelName) {}
	~ImageProcessorKernel() {}
public:
	//must be called before Run() TODO add check inside run
	virtual void Init(cl::Program& program, CImg::CImg<CIMG_TYPE>& _InputImage, CImg::CImg<CIMG_TYPE>& _OutputImage, 
		cl::CommandQueue& _Queue, cl::Buffer& _DeviceImage, cl::Buffer& _HistogramA, cl::Buffer& _HistogramB, int _num_bins, int _workgroup_size, bool _ignoreColour, bool _displayHistograms)
	{
		Image = &_DeviceImage;
		HistogramA = &_HistogramA;
		HistogramB = &_HistogramB;
		InputImage = &_InputImage;
		OutputImage = &_OutputImage;
		Queue = &_Queue;
		num_bins = _num_bins;
		workgroup_size = _workgroup_size;
		ignoreColour = _ignoreColour;
		displayHistograms = _displayHistograms;
	}
public:
	//Publicly accessible functions
	virtual void Run(bool print) = 0;
	const std::string_view GetName() const { return kernelName; }
protected:
	//references to external stuff that get re-used across different kernel runs
	int num_bins;
	int workgroup_size;
	bool ignoreColour;
	bool displayHistograms;
	cl::Buffer* Image;
	cl::Buffer* HistogramA;
	cl::Buffer* HistogramB;
	cl::CommandQueue* Queue;
	CImg::CImg<CIMG_TYPE>* InputImage;
	CImg::CImg<CIMG_TYPE>* OutputImage;
	std::string kernelName;

	//helper function to display intermediate histogram
	void ShowHistogram(const char* title) {
		if (!displayHistograms) return;
		CImg::CImg<HIST_TYPE> histDisplay(num_bins, 1, 1, 1);
		HIST_TYPE* Buf = &histDisplay.data()[0];
		Queue->enqueueReadBuffer(*HistogramA, CL_TRUE, 0, num_bins * sizeof(HIST_TYPE), Buf);
		std::filesystem::create_directory("graphs");
		std::ofstream HStream(std::string("graphs/") + std::string(title) + ".csv");
		HIST_TYPE maxVal = 0;
		for (int i = 0; i < num_bins; i++) {
			HStream << Buf[i] << ",";
			if (Buf[i] > maxVal) {
				maxVal = Buf[i];
			}
		}
		HStream.close();
		CImg::CImgDisplay disp;
		CImg::CImg<double> screenshotter;
		histDisplay._display_graph(disp, title, 1, 1, "bin", 0, num_bins, "freq", 0, (double)maxVal);
		disp.snapshot(screenshotter);
		screenshotter.save_bmp((std::string("graphs/") + std::string(title) + std::string(".bmp")).c_str());
	}
	
public:
	//(filling out rule of 5)
	//assuming defaults will be fine 
	
	//ImageProcessorKernel(const ImageProcessorKernel& other) = delete;//copy constructor
	//ImageProcessorKernel(ImageProcessorKernel&& other) noexcept = delete;//move constructor
	//ImageProcessorKernel& operator=(const ImageProcessorKernel& other) = delete;//copy assignment operator
	//ImageProcessorKernel& operator=(ImageProcessorKernel&& other) noexcept = delete;//move assignment operator
};


//derivation of ImageProcessorKernel that sets up the kernel for the basic version of the algorithm
template<typename CIMG_TYPE>
class GlobalKernel : public ImageProcessorKernel<CIMG_TYPE>
{
public:
	GlobalKernel() : ImageProcessorKernel<CIMG_TYPE>("Basic (Global)") {}
	virtual ~GlobalKernel() {}
protected:
	//kernels of each algorithm step
	cl::Kernel histogramKernel;
	cl::Kernel accumulateKernel;
	cl::Kernel normalizeKernel;
	cl::Kernel lookupKernel;
	//events to profile execution time
	cl::Event histogramEvent;
	cl::Event accumulateEvent;
	cl::Event normalizeEvent;
	cl::Event lookupEvent;
	cl::Event inputCopyEvent;
	cl::Event outputCopyEvent;
public:
	//must be called before Run()
	virtual void Init(cl::Program& program, CImg::CImg<CIMG_TYPE>& InputImage, CImg::CImg<CIMG_TYPE>& OutputImage,
		cl::CommandQueue& Queue, cl::Buffer& Image, cl::Buffer& HistogramA, cl::Buffer& HistogramB, int num_bins, int workgroup_size, bool ignoreColour, bool displayHistograms) override
	{
		ImageProcessorKernel<CIMG_TYPE>::Init(program, InputImage, OutputImage, Queue, Image, HistogramA, HistogramA, num_bins,workgroup_size, ignoreColour, displayHistograms);
		histogramKernel = cl::Kernel(program, "createHistogram_Global");
		histogramKernel.setArg(0, Image);
		histogramKernel.setArg(1, HistogramA);


		accumulateKernel = cl::Kernel(program, "AccumulateHistogram_SingleGroup");
		accumulateKernel.setArg(0, HistogramA);
		accumulateKernel.setArg(1, HistogramB);

		normalizeKernel = cl::Kernel(program, "NormalizeHistogram_Global");
		normalizeKernel.setArg(0, HistogramA);

		lookupKernel = cl::Kernel(program, "ApplyHistogram_Global");
		lookupKernel.setArg(0, Image);
		lookupKernel.setArg(1, HistogramA);
	}

	//Publicly accessible functions
	virtual void Run(bool print) override {
		//templating sucks -- idk if this issue is MSVC specific but apparently for everything i want to access from base class i have to add this
		auto Queue = this->Queue;
		auto Image = this->Image;
		auto InputImage = this->InputImage;
		auto HistogramA = this->HistogramA;
		auto HistogramB = this->HistogramB;
		auto OutputImage = this->OutputImage;
		auto ignoreColour = this->ignoreColour;
		auto num_bins = this->num_bins;
		auto workgroup_size = this->workgroup_size;


		cl_ulong kernelTotalTime = 0;
		//allowing for 2 different handlings of colour images
		int targetSpectrum = ignoreColour ? 1 : InputImage->spectrum();
		size_t imageSize = InputImage->size() / targetSpectrum;
		//adding to the global size to make sure the number of workgroups is valid
		//the kernel code ensures extra threads are skipped to prevent out-of-range memory accesses
		int imageExtraThreads = workgroup_size - (imageSize % workgroup_size);
		int histExtraThreads = workgroup_size - (num_bins % workgroup_size);

		Queue->enqueueWriteBuffer(*Image, CL_TRUE, 0, InputImage->size() * sizeof(CIMG_TYPE), &InputImage->data()[0], nullptr, &inputCopyEvent);
		for (int col = 0; col < targetSpectrum; col++) {
			//clear histograms
			Queue->enqueueFillBuffer<HIST_TYPE>(*HistogramA, 0, 0, num_bins * sizeof(HIST_TYPE));
			Queue->enqueueFillBuffer<HIST_TYPE>(*HistogramB, 0, 0, num_bins * sizeof(HIST_TYPE));
			//run kernels --offset to run each colour separately
			Queue->enqueueNDRangeKernel(histogramKernel, col * imageSize, cl::NDRange(imageSize + imageExtraThreads), cl::NDRange(workgroup_size), nullptr, &histogramEvent);
			this->ShowHistogram("GlobalBaseHistogram");
			Queue->enqueueNDRangeKernel(accumulateKernel, cl::NullRange, num_bins + histExtraThreads, cl::NDRange(workgroup_size), nullptr, &accumulateEvent);
			this->ShowHistogram("GlobalCumulativeHistogram");
			Queue->enqueueNDRangeKernel(normalizeKernel, cl::NullRange, num_bins + histExtraThreads, cl::NDRange(workgroup_size), nullptr, &normalizeEvent);
			this->ShowHistogram("GlobalNormalHistogram");
			Queue->enqueueNDRangeKernel(lookupKernel, col * imageSize, cl::NDRange(imageSize + imageExtraThreads), cl::NDRange(workgroup_size), nullptr, &lookupEvent);

			if (!print) continue;
			lookupEvent.wait();
			for (const cl::Event& event : { histogramEvent,accumulateEvent,normalizeEvent,lookupEvent }) {
				kernelTotalTime += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			}
		}


		Queue->enqueueReadBuffer(*Image, CL_TRUE, 0, OutputImage->size() * sizeof(CIMG_TYPE), &OutputImage->data()[0], nullptr, &outputCopyEvent);
		if (!print) return;
		
		
		std::cout
			<< "Copy host-to-device time [ns]: "
			<< inputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "\n"
			<< "Kernel execution time [ns]: "
			<< kernelTotalTime << "\n"
			<< "Copy device-to-host time [ns]: "
			<< outputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			<< std::endl;
	}

};

//local DECLARATION
template<typename CIMG_TYPE>
class LocalKernel : public ImageProcessorKernel<CIMG_TYPE>
{
public:
	LocalKernel() : ImageProcessorKernel<CIMG_TYPE>("Advanced (Local)") {}
	virtual ~LocalKernel() {}
protected:
	//kernels of each algorithm step
	cl::Kernel histogramKernel;
	cl::Kernel accumulate1Kernel;
	cl::Kernel accumulate2Kernel;
	cl::Kernel normalizeKernel;
	cl::Kernel lookupKernel;
	//events to profile execution time
	cl::Event histogramEvent;
	cl::Event accumulate1Event;
	cl::Event accumulate2Event;
	cl::Event normalizeEvent;
	cl::Event lookupEvent;
	cl::Event inputCopyEvent;
	cl::Event outputCopyEvent;
public:
	//must be called before Run()
	virtual void Init(cl::Program& program, CImg::CImg<CIMG_TYPE>& InputImage, CImg::CImg<CIMG_TYPE>& OutputImage,
		cl::CommandQueue& Queue, cl::Buffer& Image, cl::Buffer& HistogramA, cl::Buffer& HistogramB, int num_bins,int workgroup_size, bool ignoreColour, bool displayHistograms) override
	{
		ImageProcessorKernel<CIMG_TYPE>::Init(program, InputImage, OutputImage, Queue, Image, HistogramA, HistogramA, num_bins,workgroup_size, ignoreColour, displayHistograms);

		histogramKernel = cl::Kernel(program, "createHistogram");
		histogramKernel.setArg(0, Image);
		histogramKernel.setArg(1, HistogramA);
		histogramKernel.setArg(2, cl::Local(sizeof(HIST_TYPE) * num_bins));

		accumulate1Kernel = cl::Kernel(program, "AccumulateHistogram_1");
		accumulate1Kernel.setArg(0, HistogramA);
		accumulate1Kernel.setArg(1, HistogramB);
		accumulate1Kernel.setArg(2, cl::Local(sizeof(HIST_TYPE) * workgroup_size));
		accumulate1Kernel.setArg(3, cl::Local(sizeof(HIST_TYPE) * workgroup_size));

		accumulate2Kernel = cl::Kernel(program, "AccumulateHistogram_2");
		accumulate2Kernel.setArg(0, HistogramB);
		accumulate2Kernel.setArg(1, HistogramA);

		normalizeKernel = cl::Kernel(program, "NormalizeHistogram");
		normalizeKernel.setArg(0, HistogramA);

		lookupKernel = cl::Kernel(program, "ApplyHistogram");
		lookupKernel.setArg(0, Image);
		lookupKernel.setArg(1, HistogramA);
	}
public:
	//Publicly accessible functions
	virtual void Run(bool print) override
	{
		//templating sucks -- idk if this issue is MSVC specific but apparently for everything i want to access from base class i have to add this
		auto Queue = this->Queue;
		auto ImageBuffer = this->Image;
		auto InputImage = this->InputImage;
		auto HistogramA = this->HistogramA;
		auto HistogramB = this->HistogramB;
		auto OutputImage = this->OutputImage;
		auto ignoreColour = this->ignoreColour;
		auto num_bins = this->num_bins;
		auto workgroup_size = this->workgroup_size;

		cl_ulong kernelTotalTime = 0;
		//allowing for 2 different handlings of colour images
		int targetSpectrum = ignoreColour ? 1 : InputImage->spectrum();
		size_t imageSize = InputImage->size() / targetSpectrum;
		//adding to the global size to make sure the number of workgroups is valid
		//the kernel code ensures extra threads are skipped to prevent out-of-range memory accesses
		int imageExtraThreads = workgroup_size - (imageSize % workgroup_size);
		int histExtraThreads = workgroup_size - (num_bins % workgroup_size);
		Queue->enqueueWriteBuffer(*ImageBuffer, CL_TRUE, 0, InputImage->size() * sizeof(CIMG_TYPE), &InputImage->data()[0], nullptr, &inputCopyEvent);//initial copy
		for (int col = 0; col < targetSpectrum; col++) {
			//clear hist
			Queue->enqueueFillBuffer<HIST_TYPE>(*HistogramA, 0, 0, num_bins * sizeof(HIST_TYPE));
			Queue->enqueueFillBuffer<HIST_TYPE>(*HistogramB, 0, 0, num_bins * sizeof(HIST_TYPE));

			//run kernels -- offset so that each colour runs separately
			Queue->enqueueNDRangeKernel(histogramKernel, col * imageSize, cl::NDRange(imageSize + imageExtraThreads), cl::NDRange(workgroup_size), nullptr, &histogramEvent);
			this->ShowHistogram("LocalBaseHistogram");
			Queue->enqueueNDRangeKernel(accumulate1Kernel, cl::NullRange, num_bins + histExtraThreads, cl::NDRange(workgroup_size), nullptr, &accumulate1Event);
			Queue->enqueueNDRangeKernel(accumulate2Kernel, cl::NullRange, num_bins + histExtraThreads, cl::NDRange(workgroup_size), nullptr, &accumulate2Event);
			this->ShowHistogram("LocalCumulativeHistogram");
			Queue->enqueueNDRangeKernel(normalizeKernel, cl::NullRange, num_bins + histExtraThreads, cl::NDRange(workgroup_size), nullptr, &normalizeEvent);
			this->ShowHistogram("LocalNormalHistogram");
			Queue->enqueueNDRangeKernel(lookupKernel, col * imageSize, cl::NDRange(imageSize + imageExtraThreads), cl::NDRange(workgroup_size), nullptr, &lookupEvent);
			if (!print) continue;
			lookupEvent.wait();//this does mean runs with profiling will be somewhat slower but its not measured
			for (const cl::Event& event : { histogramEvent,accumulate1Event,accumulate2Event,normalizeEvent,lookupEvent }) {
				kernelTotalTime += event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			}
		}
		Queue->enqueueReadBuffer(*ImageBuffer, CL_TRUE, 0, OutputImage->size() * sizeof(CIMG_TYPE), &OutputImage->data()[0], nullptr, &outputCopyEvent);
		if (!print) return;
		std::cout
			<< "Copy host-to-device time [ns]: "
			<< inputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "\n"
			<< "Kernel execution time [ns]: "
			<< kernelTotalTime << "\n"
			<< "Copy device-to-host time [ns]: "
			<< outputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			<< std::endl;
	}
};

