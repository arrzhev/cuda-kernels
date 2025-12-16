#ifndef GRAYSCALE_C
#define GRAYSCALE_C

void rgb2grayInterleaved(const unsigned char* src_h, unsigned char* dst_h, unsigned size);
void rgb2grayPlanar(const unsigned char* src_h, unsigned char* dst_h, unsigned size);

#endif // GRAYSCALE_C