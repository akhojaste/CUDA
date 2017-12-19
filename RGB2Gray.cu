///The following functions transforms an RGB image into a Gray scale image using CUDA

void __global__ rgb2gray(const float *pfimagIn, float *pfimgOut, const int iWidth, const int iHeight);

void ReadAndProcessImage();

int main()
{
  //Here we call the rgb2gray function
  ReadAndProcessImage();
  
  std::cin.get();

}

void ReadAndProcessImage()
{
  int iWidth = 256;
  int iHeight = 256;
  
  float *h_ImagIn = new float[iWidth * iHeight];
  float *h_ImagOut = new float[iWidth * iHeight];
  
  //I need to get the values of the input filled with something
  
  float *d_ImagIn;
  float *d_imagOut;
  
  //Allocate memory in GPU
  cudaMemAlloc(d_ImagIn, iWidth * iHeight * sizeof(float));
  cudaMemAlloc(d_ImagOut, iWidth * iHeight * sizeof(float));
  
  //Transfer data to GPU
  cudaMemcpy(d_ImagIn, h_ImagIn, iWidth * iHeight * sizeof(float), cudaMemcpyHosttoDevice);
  
  //Compue the results in GPU
  rgb2gray(d_ImagIn, d_ImgOut, iWidth, iHeight);
  
  //Transfer data back from GPU to CPU
  cudaMemcpy(h_ImagOut, d_ImagOut, iWidth * iHeight * sizeof(float), cudaMemcpyDevicetoHost);
  
  delete pfImagIn;
  delete pfImagOut;
  
  cudaFree(d_ImagIn);
  cudaFree(d_ImagOut);
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
