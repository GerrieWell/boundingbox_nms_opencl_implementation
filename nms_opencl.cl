
// 为什么是这个数？
// 因为 1. 用位标志做mask， 所以用最长的基础类型； 2. 要32的倍数；

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}


//todo copy cuda argument
__kernel void nms_kernel(int n_boxes, float nms_overlap_thresh, 
		__global float* d_boxes, __global unsigned long *d_mask, __local float *local_boxes){
//*
	int const threadsPerBlock = sizeof(unsigned long) * 8;
	// 其实可以用 getGroupId
	//group index 
	const int row_start = get_global_id(1);
	//block col index / 当前box 对应哪【64】个boxes组的结果, col_start 可以索引哪[64]个boxes组
	const int col_start = get_global_id(0)/threadsPerBlock;
	// printf("row_start:%d \n", row_start);
	// if (row_start > col_start) return;
//copy to global for debuging
	const int row_size =
	    min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
	const int col_size =
	    min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

	// 同一个block 的所有thread 的boxes
	int item_idx = get_local_id(0);
	if (item_idx < col_size) {
	local_boxes[item_idx * 4+ 0] =
	    d_boxes[(threadsPerBlock * col_start + item_idx) * 4 + 0];
	local_boxes[item_idx * 4 + 1] =
	    d_boxes[(threadsPerBlock * col_start + item_idx) * 4 + 1];
	local_boxes[item_idx * 4 + 2] =
	    d_boxes[(threadsPerBlock * col_start + item_idx) * 4 + 2];
	local_boxes[item_idx * 4 + 3] =
	    d_boxes[(threadsPerBlock * col_start + item_idx) * 4+ 3];

	}
    barrier(CLK_LOCAL_MEM_FENCE);

	const int cur_box_idx = threadsPerBlock * row_start + item_idx;
	const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
	if (item_idx < row_size) {

		int i = 0;
	    //每个线程虽然运行的是同一段代码，但看到的表示身份的threadIdx.x是不一样的，row_size值也不一样
	    //具体的说，有188*188个线程看到的threadIdx.x是一样的，因为总共有188*188个block，threadIdx.x的取值范围是0到63

		unsigned long t = 0;
		float cur_box[4];
		for(i = 0; i < 4; i ++)
			cur_box[i] = d_boxes[cur_box_idx * 4 + i];
		int start = 0;
		if (row_start == col_start) {
			start = item_idx + 1;
		}
		const unsigned long bit1 = 1;		//不能用 long long 64位以上保留字段
		for (i = start; i < col_size; i++) {
			float iter_local_boxes[4];
			iter_local_boxes[0] = local_boxes[i * 4];
			iter_local_boxes[1] = local_boxes[i * 4 + 1];
			iter_local_boxes[2] = local_boxes[i * 4 + 2];
			iter_local_boxes[3] = local_boxes[i * 4 + 3];
		  	if (devIoU(cur_box, iter_local_boxes) > nms_overlap_thresh) {
		    	t |= (1UL << i); //？？如果用ULL 就报错。
		  	}
		}
		d_mask[cur_box_idx * col_blocks + col_start] = t;
		// int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
		// d_mask[id] = t;
    	// barrier(CLK_LOCAL_MEM_FENCE);
	}else{
		d_mask[cur_box_idx * col_blocks + col_start] = 0;
	}
	// unsigned long long t = 4;
	// int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	// d_mask[id] = t;
}