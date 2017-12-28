#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include "iostream"
#include <string>
#include <helper_timer.h>

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

	std::cout << "Total Execution time on GPU... " << timer->getTime() << " ms" << std::endl;

	namedWindow("GrayScale image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("GrayScale image", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	//system("pause");
}

void ImageProcessingCPU(Mat image)
{
	////Test the CPU

	int iCn = image.channels();
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			//Images in OpenCV are stored in the row order,i.e. row 1, row 2, ... .
			unsigned char B = image.data[i * image.cols * iCn + j * iCn + 0];
			unsigned char G = image.data[i * image.cols * iCn + j * iCn + 1];
			unsigned char R = image.data[i * image.cols * iCn + j * iCn + 2];
			
			image.data[i * image.cols * iCn + j * iCn + 0] = (B + G + R) / 3;
			image.data[i * image.cols * iCn + j * iCn + 1] = (B + G + R) / 3;
			image.data[i * image.cols * iCn + j * iCn + 2] = (B + G + R) / 3;
		}
	}
}