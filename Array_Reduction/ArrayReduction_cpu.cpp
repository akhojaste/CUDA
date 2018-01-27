#include "iostream"
using namespace std;


//The function that is defined in the CUDA
extern "C"
void Array_Reduction(float *pData, unsigned int ArraySize, long &lSum);

int main()
{
	unsigned int ArraySize = 1024 * 1024;
	float *pArray = new float[ArraySize];
	for (int i = 0; i < ArraySize; ++i)
	{
		pArray[i] = 2;
	}

	long lSum = 0;
	Array_Reduction(pArray, ArraySize, lSum);

	std::cout << "ArraySum is: " << lSum << std::endl;

	delete pArray;

	system("pause");
}
