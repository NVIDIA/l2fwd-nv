/* 
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved. 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "common.h"
#include "cuda_related.h"
#include <rte_ether.h>
#include <rte_gpudev.h>

// #define DEBUG_PRINT 1

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

/////////////////////////////////////////////////////////////////////////////////////////
//// Regular CUDA kernel -w 2
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_mac_update(struct rte_gpu_comm_list *comm_list, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t temp;
	unsigned long long pkt_start;
	
	if (idx < comm_list->num_pkts && comm_list->pkt_list[idx].addr != 0) {
		if(wtime_n)
			pkt_start = __globaltimer();

		struct rte_ether_hdr *eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list->pkt_list[idx].addr)));
		uint16_t *src_addr = (uint16_t *) (&eth->src_addr);
		uint16_t *dst_addr = (uint16_t *) (&eth->dst_addr);

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		uint8_t *src = (uint8_t *) (&eth->src_addr);
		uint8_t *dst = (uint8_t *) (&eth->dst_addr);
		printf
		    ("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
		     src[0], src[1], src[2], src[3], src[4], src[5], dst[0],
		     dst[1], dst[2], dst[3], dst[4], dst[5]);
#endif

		/* MAC update */
		temp = dst_addr[0];
		dst_addr[0] = src_addr[0];
		src_addr[0] = temp;
		temp = dst_addr[1];
		dst_addr[1] = src_addr[1];
		src_addr[1] = temp;
		temp = dst_addr[2];
		dst_addr[2] = src_addr[2];
		src_addr[2] = temp;

		if(wtime_n)
			while((__globaltimer() - pkt_start) < wtime_n);
#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		printf("After Swap, Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
				((uint8_t *) (src_addr))[0], ((uint8_t *) (src_addr))[1],
				((uint8_t *) (src_addr))[2], ((uint8_t *) (src_addr))[3],
				((uint8_t *) (src_addr))[4], ((uint8_t *) (src_addr))[5],
				((uint8_t *) (dst_addr))[0], ((uint8_t *) (dst_addr))[1],
				((uint8_t *) (dst_addr))[2], ((uint8_t *) (dst_addr))[3],
				((uint8_t *) (dst_addr))[4], ((uint8_t *) (dst_addr))[5]);
#endif
	}

	__threadfence();
	__syncthreads();

	if (idx == 0) {
		RTE_GPU_VOLATILE(*(comm_list->status_d)) = RTE_GPU_COMM_LIST_DONE;
		__threadfence_system();
	}
	__syncthreads();
}

void workload_launch_gpu_processing(struct rte_gpu_comm_list * comm_list, uint64_t wtime_n,
							int cuda_blocks, int cuda_threads, cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);

	if (comm_list == NULL)
		return;

	CUDA_CHECK(cudaGetLastError());
	kernel_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (comm_list, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////////////////
//// Persistent CUDA kernel -w 3
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_persistent_mac_update(struct rte_gpu_comm_list * comm_list, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int item_index = 0;
	unsigned long long pkt_start;
	struct rte_ether_hdr *eth;
	uint16_t *src_addr, *dst_addr, temp;
	uint32_t wait_status;
	__shared__ uint32_t wait_status_shared[1];
		
	__syncthreads();

	while (1) {
		if (idx == 0)
		{
			while (1)
			{
				wait_status = RTE_GPU_VOLATILE(comm_list[item_index].status_d[0]);
				if(wait_status != RTE_GPU_COMM_LIST_FREE)
				{
					wait_status_shared[0] = wait_status;
					__threadfence_block();
					break;
				}
			}
		}

		__syncthreads();

		if (wait_status_shared[0] != RTE_GPU_COMM_LIST_READY)
			break;

		if (idx < comm_list[item_index].num_pkts && comm_list->pkt_list[idx].addr != 0) {
			if(wtime_n)
				pkt_start = __globaltimer();

			eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list[item_index].pkt_list[idx].addr)));
			src_addr = (uint16_t *) (&eth->src_addr);
			dst_addr = (uint16_t *) (&eth->dst_addr);

#ifdef DEBUG_PRINT
			/* Code to verify source and dest of ethernet addresses */
			uint8_t *src = (uint8_t *) (&eth->src_addr);
			uint8_t *dst = (uint8_t *) (&eth->dst_addr);
			printf("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
				src[0], src[1], src[2], src[3], src[4], src[5],
				dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
#endif
			temp = dst_addr[0];
			dst_addr[0] = src_addr[0];
			src_addr[0] = temp;
			temp = dst_addr[1];
			dst_addr[1] = src_addr[1];
			src_addr[1] = temp;
			temp = dst_addr[2];
			dst_addr[2] = src_addr[2];
			src_addr[2] = temp;

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		printf("After Swap, Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
		       ((uint8_t *) (src_addr))[0], ((uint8_t *) (src_addr))[1],
		       ((uint8_t *) (src_addr))[2], ((uint8_t *) (src_addr))[3],
		       ((uint8_t *) (src_addr))[4], ((uint8_t *) (src_addr))[5],
		       ((uint8_t *) (dst_addr))[0], ((uint8_t *) (dst_addr))[1],
		       ((uint8_t *) (dst_addr))[2], ((uint8_t *) (dst_addr))[3],
		       ((uint8_t *) (dst_addr))[4], ((uint8_t *) (dst_addr))[5]);
#endif
			if(wtime_n)
				while((__globaltimer() - pkt_start) < wtime_n);
		}

		__threadfence();
		__syncthreads();
		
		if (idx == 0) {
			RTE_GPU_VOLATILE(comm_list[item_index].status_d[0]) = RTE_GPU_COMM_LIST_DONE;
			__threadfence_system();
		}

		item_index = (item_index + 1) % MAX_BURSTS_X_QUEUE;
	}
}

void workload_launch_persistent_gpu_processing(struct rte_gpu_comm_list * comm_list,
						uint64_t wtime_n,
						int cuda_blocks, int cuda_threads,
						cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);
	if (comm_list == NULL)
		return;

	CUDA_CHECK(cudaGetLastError());
	kernel_persistent_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (comm_list, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////////////////
//// CUDA GRAPHS kernel -w 4
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_graphs_mac_update(struct rte_gpu_comm_list * comm_item_list, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t temp;
	unsigned long long pkt_start;
	uint32_t wait_status;
	__shared__ uint32_t wait_status_shared[1];

	if (idx == 0)
	{
		while (1)
		{
			wait_status = RTE_GPU_VOLATILE(comm_item_list->status_d[0]);
			if(wait_status != RTE_GPU_COMM_LIST_FREE)
			{
				wait_status_shared[0] = wait_status;
				__threadfence_block();
				break;
			}
		}
	}

	__syncthreads();

	if (wait_status_shared[0] != RTE_GPU_COMM_LIST_READY)
		return;

	if (idx < comm_item_list->num_pkts && comm_item_list->pkt_list[idx].addr != 0) {
		if(wtime_n)
			pkt_start = __globaltimer();

		struct rte_ether_hdr *eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_item_list->pkt_list[idx].addr)));
		uint16_t *src_addr = (uint16_t *) (&eth->src_addr);
		uint16_t *dst_addr = (uint16_t *) (&eth->dst_addr);

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		uint8_t *src = (uint8_t *) (&eth->src_addr);
		uint8_t *dst = (uint8_t *) (&eth->dst_addr);
		printf
		    ("GRAPHS before Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
		     src[0], src[1], src[2], src[3], src[4], src[5], dst[0],
		     dst[1], dst[2], dst[3], dst[4], dst[5]);
#endif

		temp = dst_addr[0];
		dst_addr[0] = src_addr[0];
		src_addr[0] = temp;
		temp = dst_addr[1];
		dst_addr[1] = src_addr[1];
		src_addr[1] = temp;
		temp = dst_addr[2];
		dst_addr[2] = src_addr[2];
		src_addr[2] = temp;

		if(wtime_n)
			while((__globaltimer() - pkt_start) < wtime_n);

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		printf("GRAPHS after Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
		       ((uint8_t *) (src_addr))[0], ((uint8_t *) (src_addr))[1],
		       ((uint8_t *) (src_addr))[2], ((uint8_t *) (src_addr))[3],
		       ((uint8_t *) (src_addr))[4], ((uint8_t *) (src_addr))[5],
		       ((uint8_t *) (dst_addr))[0], ((uint8_t *) (dst_addr))[1],
		       ((uint8_t *) (dst_addr))[2], ((uint8_t *) (dst_addr))[3],
		       ((uint8_t *) (dst_addr))[4], ((uint8_t *) (dst_addr))[5]);
#endif
	}

	__threadfence();
	__syncthreads();

	if (idx == 0) {
		RTE_GPU_VOLATILE(*(comm_item_list->status_d)) = RTE_GPU_COMM_LIST_DONE;
		__threadfence_system();
	}
	__syncthreads();
}

void workload_launch_gpu_graph_processing(struct rte_gpu_comm_list * bitem, uint64_t wtime_n,
										int cuda_blocks, int cuda_threads, cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);

	CUDA_CHECK(cudaGetLastError());
	kernel_graphs_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (bitem, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}