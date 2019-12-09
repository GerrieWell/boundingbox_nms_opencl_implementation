#include "nms_opencl.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


#include <iostream>
#include <fstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "Cpp_common/cl.hpp"
#include "Cpp_common/util.hpp"
#include "Cpp_common/err_code.h"


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif
#define dbgl(...) {printf("line :%d\n",__LINE__);}while(0)
#define dbgv(v) {printf("%s \t:%08x %d \n",#v,v , v);}while(0)

void nms_cl_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh) ;
int nms_opencl(py::array_t<int> keep_out,py::array_t<float> boxes_host);
int opencl_context_init();


#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(cl_ulong) * 8;

int nms_opencl(py::array_t<int> keep_out,py::array_t<float> boxes_host){
    printf("nms_opencl");
    py::buffer_info boxes_host_buf = boxes_host.request();
    int boxes_num=boxes_host_buf.shape[0];
    int boxes_dim=boxes_host_buf.shape[1];
   
    // py::array_t<int>  keep_out=py::array_t<int>(boxes_num);
    py::buffer_info keep_out_buf = keep_out.request();

    float nms_overlap_thresh=0.7;
    int num_out=0;

    nms_cl_compute((int*)keep_out_buf.ptr,&num_out,(float*)boxes_host_buf.ptr,boxes_num,boxes_dim,nms_overlap_thresh);
    
    return  num_out;
}

int opencl_context_init(){
    unsigned int nx, ny;
    unsigned int iterations;

    // Create OpenCL context, queue and program
    try
    {
        cl::Context context(DEVICE);
        cl::CommandQueue queue(context);
        cl::Program program(context, util::loadProgram("./nms_opencl.cl"));
        try
        {
            program.build();
        }
        catch (cl::Error error)
        {
            // If it was a build error then show the error
            if (error.err() == CL_BUILD_PROGRAM_FAILURE)
            {
                std::vector<cl::Device> devices;
                devices = context.getInfo<CL_CONTEXT_DEVICES>();
                std::string built = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
                std::cerr << built << "\n";
            }
            throw error;
        }
    } catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ":\n";
        err_code(err.err());
        return EXIT_FAILURE;
    }        

}

// void opencl_program_init(float* boxes_host, int boxes_num,
//           int boxes_dim, float nms_overlap_thresh){

// }
// 


void nms_cl_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh) {
    printf("nms_cl_compute boxes_num :%d , boxes_dim:%d, nms_overlap_thresh %f, sizeof boxes: %d\n"
            , boxes_num, boxes_dim, nms_overlap_thresh, sizeof(boxes_host));
    std::vector<float> h_v_boxes(boxes_host, boxes_host +  boxes_num * boxes_dim * sizeof(float));
    printf("copy done size: %d \n", h_v_boxes.size());
    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
    // opencl_context_init();
    try{
        cl_int err;
        //init 
        cl::Context context(DEVICE);
        cl::CommandQueue queue(context);
        cl::Program program(context, util::loadProgram("./nms_opencl.cl"));
        try
        {
            program.build();
        }
        catch (cl::Error error)
        {
            // If it was a build error then show the error
            if (error.err() == CL_BUILD_PROGRAM_FAILURE)
            {
                std::vector<cl::Device> devices;
                devices = context.getInfo<CL_CONTEXT_DEVICES>();
                std::string built = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
                std::cerr << built << "\n";
            }
            throw error;
        }

        // boxes_dim = 4
        int work_group_dim_x_num = DIVUP(boxes_num, threadsPerBlock);
        cl::NDRange global(DIVUP(boxes_num, threadsPerBlock) * threadsPerBlock,DIVUP(boxes_num, threadsPerBlock));
        // work-group 内是一维的数据；
        cl::NDRange local_items(threadsPerBlock, 1 );
        cl::make_kernel
            <int, float, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>
            nms_kernel(program, "nms_kernel");
        int box_buffer_size =  boxes_num * boxes_dim * sizeof(float);
        dbgv(box_buffer_size);
        cl::Buffer d_boxes(context, CL_MEM_READ_WRITE,box_buffer_size);
        dbgv(h_v_boxes.end() - h_v_boxes.begin());
        // cl::copy(queue, h_v_boxes.begin(), h_v_boxes.end(), d_boxes);  // 不可用，segmentation fault 原因未知
        // cl_mem d_boxes_mem = clCreateBuffer(context(), CL_MEM_READ_WRITE, box_buffer_size, NULL, &err);
        // cl::Buffer d_boxes(d_boxes_mem);
        queue.enqueueWriteBuffer(d_boxes, CL_TRUE, 0, box_buffer_size, boxes_host, NULL, NULL);

        cl::Buffer d_mask(context, CL_MEM_READ_WRITE, boxes_num * col_blocks * sizeof(cl_ulong));

        cl::LocalSpaceArg local_boxes = cl::Local(sizeof(float) * threadsPerBlock * 4);
        cl::EnqueueArgs args(queue, global, local_items);
        nms_kernel(args, boxes_num, nms_overlap_thresh, d_boxes, d_mask, local_boxes);
        std::vector<cl_ulong> h_mask(boxes_num * col_blocks );
        cl::copy(queue, d_mask, h_mask.begin(), h_mask.end());

        std::vector<cl_ulong> remv(col_blocks);
        memset(&remv[0], 0, sizeof(cl_ulong) * col_blocks);

        int num_to_keep = 0;
        for (int i = 0; i < boxes_num; i++) {
            // nth block 
            int nblock = i / threadsPerBlock;
            // ith in block
            int inblock = i % threadsPerBlock;

            if (!(remv[nblock] & (1UL << inblock))) {
                keep_out[num_to_keep++] = i;
                // cl_ulong *p = &h_mask[0] + i * col_blocks;
                // printf("%llu\n",*p);
                // for (int j = nblock; j < col_blocks; j++) {
                    // remv[j] |= p[j];
                // }
                // 当前box 和其他 block 
                for(int j = nblock; j < col_blocks;j++){
                    remv[j] |= h_mask[i * col_blocks + j];
                }
            }
        }
        *num_out=num_to_keep;
    } catch (cl::Error err)
    {
        util::print_trace();
        std::cerr << "ERROR: " << err.what() << ":\n";
        std::cerr << "error code :" << err_code(err.err())<<"@!\n";
        return ;
    }
    
}

PYBIND11_MODULE(nms_opencl, m) {
    m.doc() = "nms_opencl"; 
    m.def("nms_opencl", &nms_opencl, "A function which nms_cl");
}