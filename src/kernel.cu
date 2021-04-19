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
__global__ void kernel_mac_update(uintptr_t * addr, int num_pkts, uint32_t * status, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t temp;
	unsigned long long pkt_start;

	if (idx < num_pkts) {
		if(wtime_n)
			pkt_start = __globaltimer();
		struct rte_ether_hdr *eth = (struct rte_ether_hdr *)(((uint8_t *) (addr[idx])));
		uint16_t *src_addr = (uint16_t *) (&eth->s_addr);
		uint16_t *dst_addr = (uint16_t *) (&eth->d_addr);

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		uint8_t *src = (uint8_t *) (&eth->s_addr);
		uint8_t *dst = (uint8_t *) (&eth->d_addr);
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
		status[0] = BURST_DONE;
		__threadfence_system();
	}
	__syncthreads();
}

void workload_launch_gpu_processing(uintptr_t * addr, int num, uint32_t * status, 
								uint64_t wtime_n,
								int cuda_blocks, int cuda_threads, cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);

	if (addr == NULL)
		return;

	CUDA_CHECK(cudaGetLastError());
	kernel_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (addr, num, status, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////////////////
//// Persistent CUDA kernel -w 3
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_persistent_mac_update(struct burst_item * bitem_list, uint32_t * wait_list_d, uint64_t wtime_n)
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
				wait_status = ACCESS_ONCE(wait_list_d[item_index]);
				if(wait_status != BURST_FREE)
				{
					wait_status_shared[0] = wait_status;
					__threadfence_block();
					break;
				}
			}
		}

		__syncthreads();

		if (wait_status_shared[0] != BURST_READY)
			break;

		if (idx < bitem_list[item_index].num_mbufs) {
			if(wtime_n)
				pkt_start = __globaltimer();

			eth = (struct rte_ether_hdr *)(((uint8_t *) (bitem_list[item_index].addr[idx])));
			src_addr = (uint16_t *) (&eth->s_addr);
			dst_addr = (uint16_t *) (&eth->d_addr);

#ifdef DEBUG_PRINT
			/* Code to verify source and dest of ethernet addresses */
			uint8_t *src = (uint8_t *) (&eth->s_addr);
			uint8_t *dst = (uint8_t *) (&eth->d_addr);
			printf
			    ("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
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
			ACCESS_ONCE(bitem_list[item_index].status) = BURST_DONE;
			__threadfence_system();
		}

		item_index = (item_index + 1) % MAX_BURSTS_X_QUEUE;
	}
}

void workload_launch_persistent_gpu_processing(struct burst_item * bitem_list,
						uint32_t * wait_list_d,
						uint64_t wtime_n,
						int cuda_blocks, int cuda_threads,
						cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);
	if (bitem_list == NULL)
		return;

	CUDA_CHECK(cudaGetLastError());
	kernel_persistent_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (bitem_list, wait_list_d, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////////////////
//// CUDA GRAPHS kernel -w 4
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_graphs_mac_update(struct burst_item * bitem, uint32_t * wait_list_d, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t temp;
	unsigned long long pkt_start;
	uint32_t wait_status;
	__shared__ uint32_t wait_status_shared[1];
	struct rte_ether_hdr * eth;

	if (idx == 0)
	{
		while (1)
		{
			wait_status = ACCESS_ONCE(wait_list_d[0]);
			if(wait_status != BURST_FREE)
			{
				wait_status_shared[0] = wait_status;
				__threadfence_block();
				break;
			}
		}
	}

	__syncthreads();

	if (wait_status_shared[0] != BURST_READY)
		return;

	if (idx < bitem->num_mbufs) {
		if(wtime_n)
			pkt_start = __globaltimer();
		eth = (struct rte_ether_hdr *)(((uint8_t *) (bitem->addr[idx])));
		uint16_t *src_addr = (uint16_t *) (&eth->s_addr);
		uint16_t *dst_addr = (uint16_t *) (&eth->d_addr);

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		uint8_t *src = (uint8_t *) (&eth->s_addr);
		uint8_t *dst = (uint8_t *) (&eth->d_addr);
		printf
		    ("Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
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
		printf("2 Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
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
		ACCESS_ONCE(bitem->status) = BURST_DONE;
		__threadfence_system();
	}
	__syncthreads();
}

void workload_launch_gpu_graph_processing(struct burst_item * bitem,  uint32_t * wait_list_d, uint64_t wtime_n,
										int cuda_blocks, int cuda_threads, cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);

	CUDA_CHECK(cudaGetLastError());
	kernel_graphs_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (bitem, wait_list_d, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}