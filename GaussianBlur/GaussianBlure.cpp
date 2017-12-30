#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include "iostream"
#include <string>
#include <helper_timer.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//The GPU implementation is inside RGB2Gray.cu file
extern "C"
void ImageProcessingGPU(Mat image);

//The CPU version
void ImageProcessingCPU(Mat image);

int main()
{
	//Read an image using the OpenCV library
	string file = "bing.png";

	Mat image;
	// Read the file
	image = imread(file.c_str(), IMREAD_COLOR); 

	//change the color format to RGB instread of BGR as in OpenCV colors are reveresed.
	//cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BayerBG2RGB);

	// Check for invalid input
	if (!image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("Original Image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original Image", image); // Show our image inside it.
	
	//Creating a timer to time the performance
	StopWatchInterface *timer = nullptr;
	sdkCreateTimer(&timer);
	
	//CPU code
	//ImageProcessingCPU(image);

	std::cout << "Launching the GPU kernel code.." << std::endl;

	sdkStartTimer(&timer);
	//Launch the processing code to be executed in the GPU
	ImageProcessingGPU(image);
	sdkStopTimer(&timer);

	//std::cout << "Total Execution time on GPU... " << timer->getTime() << " ms" << std::endl;

	namedWindow("Smoothed image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Smoothed image", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	//system("pause");
}

void ImageProcessingCPU(Mat image)
{

	//A 5x5 Gaussian smotthing filter
	//const short sFilterW = 5;
	//const short sFilterH = 5;
	//const short sFilterSize = 25;
	//double dbFilterCoeff[sFilterSize] = { 0.0232468398782944, 0.0338239524399223, 0.0383275593839039, 0.0338239524399223, 0.0232468398782944,
	//								 0.0338239524399223, 0.0492135604085414, 0.0557662698468495, 0.0492135604085414, 0.0338239524399223,
	//								 0.0383275593839039, 0.0557662698468495, 0.0631914624102647, 0.0557662698468495, 0.0383275593839039,
	//								 0.0338239524399223, 0.0492135604085414, 0.0557662698468495, 0.0492135604085414, 0.0338239524399223,
	//								 0.0232468398782944, 0.0338239524399223, 0.0383275593839039, 0.0338239524399223, 0.0232468398782944 };

	//A 3x3 Sharpening filter
	const short sFilterW = 3;
	const short sFilterH = 3;
	const short sFilterSize = 9;
	double dbFilterCoeff[sFilterSize] = { -0.0833333333333333, -0.0833333333333333, -0.0833333333333333,
										  -0.0833333333333333, 1.66666666666667, -0.0833333333333333,
									      -0.0833333333333333, -0.0833333333333333, -0.0833333333333333 };




	long long lStart, lEnd, lFreq;
	QueryPerformanceCounter((LARGE_INTEGER*)&lStart);

	QueryPerformanceFrequency((LARGE_INTEGER*)&lFreq);

	BYTE *pline = nullptr;
	long lRow = 0, lCol = 0;

	unsigned short usWidth = image.cols;
	unsigned short usHeight = image.rows;
	unsigned short usChannels = image.channels();

	long lLineWidth = (long)usWidth * usChannels;

	short sFRow = 0, sFCol = 0;

	//#pragma omp parallel for //why this does not make it better? because the buffer is shared between thread and this will actually add the overhead of waiting for the buffer to get released.
	for (int i = 0; i < usHeight * usWidth; ++i)
	{

		lRow = i / usWidth;
		lCol = i %usWidth;

		double dbFilterOutputB = 0.0;
		double dbFilterOutputG = 0.0;
		double dbFilterOutputR = 0.0;

		for (int iFilterCount = 0; iFilterCount < sFilterSize; ++iFilterCount)
		{
			sFRow = iFilterCount / sFilterW;
			sFCol = iFilterCount % sFilterW;

			long lImageRow = lRow + sFRow - sFilterH / 2;
			long lImageCol = lCol + sFCol - sFilterW / 2;

			if (lImageRow < 0 || lImageRow >= usWidth || lImageCol < 0 || lImageCol >= usHeight)
				continue;

			dbFilterOutputB += dbFilterCoeff[sFRow * sFilterW + sFCol] * (float)image.data[lImageRow * usWidth * usChannels + lImageCol * usChannels]; //B
			dbFilterOutputG += dbFilterCoeff[sFRow * sFilterW + sFCol] * (float)image.data[lImageRow * usWidth * usChannels + lImageCol * usChannels + 1]; //G
			dbFilterOutputR += dbFilterCoeff[sFRow * sFilterW + sFCol] * (float)image.data[lImageRow * usWidth * usChannels + lImageCol * usChannels + 2]; //R
		}

		//The filtered values might be out of range, either be negative or bigger than 255. We can't show them.
		if (dbFilterOutputB < 0.0)
			dbFilterOutputB = 0.0;
		else if (dbFilterOutputB >= 255.0)
			dbFilterOutputB = 255.0;

		if (dbFilterOutputG < 0.0)
			dbFilterOutputG = 0.0;
		else if (dbFilterOutputG >= 255.0)
			dbFilterOutputG = 255.0;

		if (dbFilterOutputR < 0.0)
			dbFilterOutputR = 0.0;
		else if (dbFilterOutputR >= 255.0)
			dbFilterOutputR = 255.0;

		image.data[lRow * usWidth * usChannels + lCol * usChannels + 0] = dbFilterOutputB; //B
		image.data[lRow * usWidth * usChannels + lCol * usChannels + 1] = dbFilterOutputG; //G
		image.data[lRow * usWidth * usChannels + lCol * usChannels + 2] = dbFilterOutputR; //R
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&lEnd);

	double dbTime = ((double)lEnd - (double)lStart) / (double)lFreq;

	dbTime *= 1000.0; //ms

}