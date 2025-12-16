#ifndef MEAN_BLUR_C
#define MEAN_BLUR_C

void meanBlurColor(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols, int kernelSize);
void meanBlurGray(const unsigned char* src, unsigned char* dst, unsigned rows, unsigned cols, int kernelSize);

#endif // MEAN_BLUR_C