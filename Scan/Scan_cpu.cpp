#include "iostream"
using namespace std;


//The function that is defined in the CUDA
extern "C"
void Scan(float *pDataIn, float *pDataOut, unsigned int ArraySize);

int main()
{
	unsigned int ArraySize = 512;
	float *pArrayIn = new float[ArraySize];
	float *pArrayOut = new float[ArraySize];
	for (int i = 0; i < ArraySize; ++i)
	{
		pArrayIn[i] = i + 1;
	}

	Scan(pArrayIn, pArrayOut, ArraySize);

	delete pArrayIn;
	delete pArrayOut;

	system("pause");
}