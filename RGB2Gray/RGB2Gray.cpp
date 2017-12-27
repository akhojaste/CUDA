#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

#include "iostream"
#include <string>

//The GPU implementation is inside RGB2Gray.cu file
extern "C"
void ImageProcessingGPU();

int main()
{
	//Read an image using the OpenCv library
	string file = "C:\\Users\\amir\\Desktop\\Github\\ImageProcessing\\cameraman.png";

	Mat image;
	image = imread(file.c_str(), IMREAD_COLOR); // Read the file

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	std::cout << "Launching the GPU kernel code.." << std::endl;

	//Launch the processing code to be executed in the GPU
	ImageProcessingGPU();

	system("pause");
}

