#include "iostream"
#include <string>

//The GPU implementation is inside RGB2Gray.cu file
extern "C"
void ImageProcessingGPU();
unsigned char* readBMP(const char* filename);

int main()
{
	std::cout << "Launching the GPU kernel code.." << std::endl;

	//Launch the processing code to be executed in the GPU
	ImageProcessingGPU();

	system("pause");
}

