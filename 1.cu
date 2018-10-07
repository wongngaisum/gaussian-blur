#include<stdio.h>
#include<stdlib.h>

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define CREATOR "COMP3231"
#define RGB_COMPONENT_COLOR 255
#define thread_x 10
#define thread_y 10

#define CUDA_CHECK(err) (cuda_checker(err, __FILE__, __LINE__))

static void cuda_checker(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

__global__ void blur_kernel(PPMImage *dev_img, PPMPixel *out_data) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x + gridDim.x * blockDim.x * y;   // index in whole image

    float filter[] = {0.05, 0.1, 0.05, 0.1, 0.4, 0.1, 0.05, 0.1, 0.05}; // localVar filter

    // width and height of image
    int width = dev_img->x;
    int height = dev_img->y;
    __shared__ PPMPixel data[thread_y + 2][thread_x + 2];   // sharedVar pixels of image

    // empty pixel
    PPMPixel empty;
    empty.red = 0;
    empty.green = 0;
    empty.blue = 0;
    int local_idx = threadIdx.x + threadIdx.y * blockDim.x; // index in block

    // read pixels surrounding block into shared memory
    if (local_idx < blockDim.x + 2) {
        int tmp_x = blockIdx.x * blockDim.x - 1 + local_idx;
        int tmp_y = blockIdx.y * blockDim.y - 1;
        if (tmp_x < 0 || tmp_y < 0 || tmp_x >= blockDim.x * gridDim.x || tmp_y >= blockDim.y * gridDim.y)
            data [0][local_idx] = empty;
        else
            data [0][local_idx] = dev_img->data[tmp_x + tmp_y * gridDim.x * blockDim.x];

        tmp_y = (blockIdx.y + 1) * blockDim.y;   
        if (tmp_x < 0 || tmp_y < 0 || tmp_x >= blockDim.x * gridDim.x || tmp_y >= blockDim.y * gridDim.y)
            data[blockDim.y + 1][local_idx] = empty;
        else
            data[blockDim.y + 1][local_idx] = dev_img->data[tmp_x + tmp_y * gridDim.x * blockDim.x];    
    }

    if (local_idx < blockDim.y) {
        int tmp_x = blockIdx.x * blockDim.x - 1;
        int tmp_y = blockIdx.y * blockDim.y + local_idx;
        if (tmp_x < 0 || tmp_y < 0 || tmp_x >= blockDim.x * gridDim.x || tmp_y >= blockDim.y * gridDim.y)
            data[local_idx + 1][0] = empty;
        else
            data[local_idx + 1][0] = dev_img->data[tmp_x + tmp_y * gridDim.x * blockDim.x];

        tmp_x = (blockIdx.x + 1) * blockDim.x;   
        if (tmp_x < 0 || tmp_y < 0 || tmp_x >= blockDim.x * gridDim.x || tmp_y >= blockDim.y * gridDim.y)
            data[local_idx + 1][blockDim.x + 1] = empty;
        else
            data[local_idx + 1][blockDim.x + 1] = dev_img->data[tmp_x + tmp_y * gridDim.x * blockDim.x];
    }

    // read data into shared memory
    if (idx < width * height)
        data[threadIdx.y + 1][threadIdx.x + 1] = dev_img->data[idx];

    __syncthreads();

    if (idx < width * height) {    // avoid illegal memory access
        out_data[idx].red = data[threadIdx.y][threadIdx.x].red * filter[0] + data[threadIdx.y][threadIdx.x + 1].red * filter[1] + 
                                data[threadIdx.y][threadIdx.x + 2].red * filter[2] + data[threadIdx.y + 1][threadIdx.x].red * filter[3] +
                                data[threadIdx.y + 1][threadIdx.x + 1].red * filter[4] + data[threadIdx.y + 1][threadIdx.x + 2].red * filter[5] +
                                data[threadIdx.y + 2][threadIdx.x].red * filter[6] + data[threadIdx.y + 2][threadIdx.x + 1].red * filter[7] +
                                data[threadIdx.y + 2][threadIdx.x + 2].red * filter[8];

        out_data[idx].green = data[threadIdx.y][threadIdx.x].green * filter[0] + data[threadIdx.y][threadIdx.x + 1].green * filter[1] + 
                                data[threadIdx.y][threadIdx.x + 2].green * filter[2] + data[threadIdx.y + 1][threadIdx.x].green * filter[3] +
                                data[threadIdx.y + 1][threadIdx.x + 1].green * filter[4] + data[threadIdx.y + 1][threadIdx.x + 2].green * filter[5] +
                                data[threadIdx.y + 2][threadIdx.x].green * filter[6] + data[threadIdx.y + 2][threadIdx.x + 1].green * filter[7] +
                                data[threadIdx.y + 2][threadIdx.x + 2].green * filter[8];

        out_data[idx].blue = data[threadIdx.y][threadIdx.x].blue * filter[0] + data[threadIdx.y][threadIdx.x + 1].blue * filter[1] + 
                                data[threadIdx.y][threadIdx.x + 2].blue * filter[2] + data[threadIdx.y + 1][threadIdx.x].blue * filter[3] +
                                data[threadIdx.y + 1][threadIdx.x + 1].blue * filter[4] + data[threadIdx.y + 1][threadIdx.x + 2].blue * filter[5] +
                                data[threadIdx.y + 2][threadIdx.x].blue * filter[6] + data[threadIdx.y + 2][threadIdx.x + 1].blue * filter[7] +
                                data[threadIdx.y + 2][threadIdx.x + 2].blue * filter[8];
    }
}

void your_gaussian_blur_func(PPMImage *img) {

    PPMImage *host_img; // for assigning PPMPixel pointer on device

    host_img = (PPMImage *) malloc(sizeof(PPMImage));
    memcpy(host_img, img, sizeof(PPMImage));

    CUDA_CHECK(cudaMalloc((void**)&(host_img->data), img->x * img->y * sizeof(PPMPixel)));  // allocate PPMPixel pointer on device
    CUDA_CHECK(cudaMemcpy(host_img->data, img->data, img->x * img->y * sizeof(PPMPixel), cudaMemcpyHostToDevice));  // copy PPMPixel data to device

    // PPMPixel data is now on the gpu, now copy the "meta" data to gpu

    PPMImage *dev_img;  // for assigning PPMImage on device
    CUDA_CHECK(cudaMalloc((void**)&dev_img, sizeof(PPMImage)));  // allocate memory on device
    CUDA_CHECK(cudaMemcpy(dev_img, host_img, sizeof(PPMImage), cudaMemcpyHostToDevice));  // copy memory to device

    PPMPixel *out_data;
    CUDA_CHECK(cudaMalloc((void**)&(out_data), img->x * img->y * sizeof(PPMPixel)));  // allocate PPMPixel pointer on device

    dim3 threadsPerBlock = dim3(thread_x, thread_y);
    dim3 blocksPerGrid = dim3((img->x + thread_x - 1) / thread_x, (img->y + thread_y - 1) / thread_y);
    
    blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_img, out_data);

    CUDA_CHECK(cudaMemcpy(img->data, out_data, img->x * img->y * sizeof(PPMPixel), cudaMemcpyDeviceToHost));  // copy memory to host

    CUDA_CHECK(cudaFree(out_data));
    CUDA_CHECK(cudaFree(host_img->data));
    CUDA_CHECK(cudaFree(dev_img));
    free(host_img);
}

int main(){
    // read
    PPMImage *image;
    image = readPPM("input.ppm");

    // record execution time
    float time;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    your_gaussian_blur_func(image);

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

    printf("Time to generate:  %3.1f ms \n", time);   

    // write
    writePPM("output.ppm",image);
}