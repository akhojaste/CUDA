///The following functions transforms an RGB image into a Gray scale image using CUDA

void __global__ rgb2gray(const float *pfimagIn, float *pfimgOut, const int iWidth, const int iHeight);


int main()
{

  //Here we call the rgb2gray function
  
  std::cin.get();

}

void __global__ rgb2gray(const float *pfimagIn, float *pfimgOut, const int iWidth, const int iHeight)
{
  
  int iRow = blockIdx.y * blockDim.y + threadIdx.y;
  int iCol = blockIdx.x + blockDim.x + threadIdx.x;

  float fPixel = pfImagIn[iRow * iWidth + iCol];
  fPixel = fPixel & (0x000000FF); // Just get the R value; image format is ARGB

  float fPixelOut = fPixel & (fPixel << 8) & (fPixel << 16);;
  fPixelOut = fPixelOut && 0x00000000; //No Transparency in Alpha channel;
  
  pfImageOut[iRow * iWidth + iCol] = fPixelOut;
  

}
