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

#include "pipeline.h"

Pipeline::Pipeline(int _index,
					int _workload_type,
					int _pkt_time_ns,
					gdr_t * _gdr_descr,
					int _rx_queue,
					int _tx_queue,
					int _gpu_id)
					:
					index(_index),
					workload_type(_workload_type),
					pkt_time_ns(_pkt_time_ns),
					gdr_descr(_gdr_descr),
					rx_queue(_rx_queue),
					tx_queue(_tx_queue),
					gpu_id(_gpu_id)
{
	comm_list = rte_gpu_comm_create_list(gpu_id, MAX_BURSTS_X_QUEUE);
	if(comm_list == NULL)
		rte_panic("rte_gpu_comm_create_list");

	start_rx_measure = false;
	start_tx_measure = false;
	rx_pkts = 0;
	tx_pkts = 0;
	rx_bytes = 0;
	tx_bytes = 0;
	start_rx_core = Time::zeroNs();
	stop_rx_core = Time::zeroNs();
	start_tx_core = Time::zeroNs();
	stop_tx_core = Time::zeroNs();
	pipeline_force_quit = 0;

	if(workload_type >= GPU_WORKLOAD)
		CUDA_CHECK(cudaStreamCreateWithFlags(&(c_stream), cudaStreamNonBlocking));

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// PERSISTENT WORKLOAD (PK or GRAPHS)
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (workload_type >= GPU_PK_WORKLOAD)
	{
		if (0 != gdrcopy_alloc_pin(
							gdr_descr,
							&(notify_kernel_list.ready_mh), 
							&(notify_kernel_list.ready_d), 
							&(notify_kernel_list.ready_h),
							&(notify_kernel_list.ready_free),
							&(notify_kernel_list.ready_size),
							MAX_BURSTS_X_QUEUE * sizeof(uint32_t))
		) {
			fprintf(stderr, "gdrcopy_alloc_pin flush failed\n");
			exit(EXIT_FAILURE);
		}

		for (int index_item = 0; index_item < MAX_BURSTS_X_QUEUE; index_item++)
			RTE_GPU_VOLATILE(((uint32_t*)(notify_kernel_list.ready_h))[index_item]) = BURST_FREE;			

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//// ONE CUDA KERNEL PER PIPELINE
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if(workload_type == GPU_PK_WORKLOAD)
		{
			workload_launch_persistent_gpu_processing(comm_list, (uint32_t*)(notify_kernel_list.ready_d),
												pkt_time_ns,
												PK_CUDA_BLOCKS, PK_CUDA_THREADS, c_stream);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//// CUDA GRAPHS
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if(workload_type == GPU_GRAPHS_WORKLOAD)
		{
			for(int index_g = 0; index_g < N_GRAPHS; index_g++)
			{
				cudaStreamBeginCapture(c_stream, cudaStreamCaptureModeGlobal);
				for(int index_b = index_g*GRAPH_BURST; index_b < ((index_g+1)*GRAPH_BURST); index_b++)
				{
					workload_launch_gpu_graph_processing(
								&(comm_list[index_b]), &(((uint32_t*)(notify_kernel_list.ready_d))[index_b]),
								pkt_time_ns, 
								MAC_CUDA_BLOCKS, MAC_THREADS_BLOCK, c_stream
							);
				}
				cudaStreamEndCapture(c_stream, &wgraph[index_g]);
				cudaGraphInstantiate(&winstance[index_g], wgraph[index_g], NULL, NULL, 0);
			}
		}
	}
}

Pipeline::~Pipeline() {

	if(workload_type >= GPU_WORKLOAD)
		cudaStreamDestroy(c_stream);
	
	rte_gpu_comm_destroy_list(comm_list, MAX_BURSTS_X_QUEUE);

	if (workload_type >= GPU_PK_WORKLOAD)
		gdrcopy_cleanup(
							*gdr_descr,
							(CUdeviceptr) notify_kernel_list.ready_free,
							notify_kernel_list.ready_mh,
							(void*)notify_kernel_list.ready_h,
							notify_kernel_list.ready_size
						);

	if(workload_type == GPU_GRAPHS_WORKLOAD)
	{
		for(int index_g = 0; index_g < N_GRAPHS; index_g++)
		{
			cudaGraphDestroy(wgraph[index_g]);
			// cudaGraphExecDestroy ?
		}	
	}

}

void Pipeline::terminateWorkload() {
	if(workload_type == GPU_PK_WORKLOAD)
	{
		printf("Killing persistent kernel...\n");
		for (int index_item = 0; index_item < MAX_BURSTS_X_QUEUE; index_item++) {
			RTE_GPU_VOLATILE(comm_list[index_item].status) = RTE_GPU_COMM_LIST_ERROR;
			RTE_GPU_VOLATILE(((uint32_t*)(notify_kernel_list.ready_h))[index_item]) = RTE_GPU_COMM_LIST_ERROR;
		}

		CUDA_CHECK(cudaStreamSynchronize(c_stream));
	}
}