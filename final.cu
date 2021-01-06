#include <cuda.h>
#include <png.h>
#include <zlib.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
//#include <cuda_runtime.h>

#define MASK_X 5
#define MASK_Y 5
#define KERNEL_RADIUS 2
#define SCALE 8

/* Hint 7 */
// this variable is used by device
__constant__ int d_Kernel[MASK_X][MASK_Y] = {{-1, -2, 0, 2, 1},
                            {-4, -8, 0, 8, 4},
                            {-6, -12, 0, 12, 6},
                            {-4, -8, 0, 8, 4},
                            {-1, -2, 0, 2, 1}};

int read_png(const char* filename, unsigned char** image, unsigned* height,
             unsigned* width, unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type,
                 NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height,
               const unsigned width, const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* Hint 5 */
// this function is called by host and executed by device
/*
__global__ void myconv(unsigned char* _host_s, unsigned char* _host_t,
                       int* _mask, unsigned height, unsigned width,
                       unsigned channels) {
    int x, y, i, v, u;
    int R, G, B;
    double val[MASK_N * 3] = {0.0};
    int adjustX, adjustY, xBound, yBound;

    // parallel job by blockIdx, blockDim, threadIdx
    for (y = blockIdx.x; y < blockIdx.x + 1; ++y) {
        // printf("this is %d\n", y);
        if (threadIdx.x >= width) break;
        for (x = threadIdx.x; x < width; x += blockDim.x) {
            for (i = 0; i < MASK_N; ++i) {
                adjustX = (MASK_X % 2) ? 1 : 0;
                adjustY = (MASK_Y % 2) ? 1 : 0;
                xBound = MASK_X / 2;
                yBound = MASK_Y / 2;

                val[i * 3 + 2] = 0.0;
                val[i * 3 + 1] = 0.0;
                val[i * 3] = 0.0;

                for (v = -yBound; v < yBound + adjustY; ++v) {
                    for (u = -xBound; u < xBound + adjustX; ++u) {
                        if ((x + u) >= 0 && (x + u) < width && y + v >= 0 &&
                            y + v < height) {
                            R = _host_s[channels * (width * (y + v) + (x + u)) +
                                        2];
                            G = _host_s[channels * (width * (y + v) + (x + u)) +
                                        1];
                            B = _host_s[channels * (width * (y + v) + (x + u)) +
                                        0];
                            val[i*3+2] += R * mask[i][u + xBound][v + yBound];
                            val[i*3+1] += G * mask[i][u + xBound][v + yBound];
                            val[i*3+0] += B * mask[i][u + xBound][v + yBound];
                            val[i * 3 + 2] +=
                                R * _mask[i * (MASK_X * MASK_Y) +
                                          (u + xBound) * MASK_X + v + yBound];
                            val[i * 3 + 1] +=
                                G * _mask[i * (MASK_X * MASK_Y) +
                                          (u + xBound) * MASK_X + v + yBound];
                            val[i * 3 + 0] +=
                                B * _mask[i * (MASK_X * MASK_Y) +
                                          (u + xBound) * MASK_X + v + yBound];
                        }
                    }
                }
            }

            double totalR = 0.0;
            double totalG = 0.0;
            double totalB = 0.0;
            for (i = 0; i < MASK_N; ++i) {
                totalR += val[i * 3 + 2] * val[i * 3 + 2];
                totalG += val[i * 3 + 1] * val[i * 3 + 1];
                totalB += val[i * 3 + 0] * val[i * 3 + 0];
            }

            totalR = sqrt(totalR) / SCALE;
            totalG = sqrt(totalG) / SCALE;
            totalB = sqrt(totalB) / SCALE;
            const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
            const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
            const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
            _host_t[channels * (width * y + x) + 2] = cR;
            _host_t[channels * (width * y + x) + 1] = cG;
            _host_t[channels * (width * y + x) + 0] = cB;
        }
    }
}
*/

__global__ void convolutionGPU(unsigned char* d_Result, unsigned char* d_Data, int dataW,
                               int dataH) {
    //////////////////////////////////////////////////////////////////////
    // most slowest way to compute convolution
    //////////////////////////////////////////////////////////////////////

    // global mem address for this thread
    const int gLoc = threadIdx.x + blockIdx.x * blockDim.x +
                     threadIdx.y * dataW + blockIdx.y * blockDim.y * dataW;

    float sum = 0;
    float value = 0;
    unsigned char output;
    for (int c = 0; c < 3; ++c) {
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)  // row wise
        {
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)  // col wise
            {
                // check row first
                if (blockIdx.x == 0 && (threadIdx.x + i) < 0)  // left apron
                    value = 0;
                else if (blockIdx.x == (gridDim.x - 1) &&
                         (threadIdx.x + i) > blockDim.x - 1)  // right apron
                    value = 0;
                else {
                    // check col next
                    if (blockIdx.y == 0 && (threadIdx.y + j) < 0)  // top apron
                        value = 0;
                    else if (blockIdx.y == (gridDim.y - 1) &&
                             (threadIdx.y + j) >
                                 blockDim.y - 1)  // bottom apron
                        value = 0;
                    else  // safe case
                        value = d_Data[c * (dataW * dataH) + gLoc + i + j * dataW];
                }
                sum += value * d_Kernel[KERNEL_RADIUS + i][KERNEL_RADIUS + j];
            }
            output = (sum > 255.0) ? 255 : sum;
            d_Result[c * (dataW * dataH) + gLoc] = output;
        }
    }
}

int main(int argc, char** argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char* host_t = (unsigned char*)malloc(height * width * channels *
                                                   sizeof(unsigned char));
    int total_bytes = height * width * channels * sizeof(unsigned char);

    // threads per blocks : 1024
    // total blocks : 2

    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst
    unsigned char *_host_s = NULL, *_host_t = NULL;
    int* _mask = NULL;
    cudaMalloc((&_host_s), total_bytes);
    cudaMalloc((&_host_t), total_bytes);
    //cudaMalloc((&_mask), MASK_X * MASK_Y * sizeof(int));

    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(_host_s, host_s, total_bytes, cudaMemcpyHostToDevice);
    /*
    cudaMemcpy(_mask, d_Kernel, MASK_X * MASK_Y * sizeof(int),
               cudaMemcpyHostToDevice);
               */

    /* Hint 3 */
    // acclerate this function
    int nBlock = height, threadPerBlock = 64;
    // myconv<<<nBlock, threadPerBlock>>>(_host_s, _host_t, _mask, height,
    // width,
    //                                   channels);
    int row_div = ceil(height / MASK_X);
    int col_div = ceil(width / MASK_Y);
    dim3 block(col_div, row_div);
    dim3 thread(MASK_X, MASK_Y);
    convolutionGPU<<<block, thread>>>(_host_s, _host_t, height,
                                               width);

    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(host_t, _host_t, total_bytes, cudaMemcpyDeviceToHost);
    write_png(argv[2], host_t, height, width, channels);

    return 0;
}
