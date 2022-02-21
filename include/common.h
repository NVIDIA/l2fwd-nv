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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <rte_mbuf.h>

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

static constexpr uint32_t RTE_LOGTYPE_L2FWD = RTE_LOGTYPE_USER1;
static constexpr uint32_t DEFAULT_PKT_BURST =  32;
static constexpr uint32_t DEFAULT_PKT_BURST_INPUT = 8192;
static constexpr uint32_t DEFAULT_MBUF_DATAROOM = 2048;
static constexpr uint32_t MAC_CUDA_BLOCKS = 1;
static constexpr uint32_t MAC_THREADS_BLOCK = 256;
static constexpr uint32_t PK_CUDA_BLOCKS = 1;
static constexpr uint32_t PK_CUDA_THREADS = 1024;

static constexpr uint32_t MAX_MBUFS_BURST = PK_CUDA_THREADS;
static constexpr uint32_t MAX_BURSTS_X_QUEUE = 4096;
static constexpr uint32_t GRAPH_BURST =	 8;
static constexpr uint32_t N_GRAPHS = MAX_BURSTS_X_QUEUE / GRAPH_BURST;

static constexpr uint32_t BURST_FREE = 0;
static constexpr uint32_t BURST_READY = 1;
static constexpr uint32_t BURST_DONE = 2;
static constexpr uint32_t BURST_EXIT = 3;
/*
 * Configurable number of RX/TX ring descriptors
 */
static constexpr uint32_t RX_DESC_DEFAULT = 1024;
static constexpr uint32_t TX_DESC_DEFAULT = 1024;
static constexpr uint32_t MAX_CORES =  16;
static constexpr uint32_t MAX_PIPELINES = 8;
static constexpr uint32_t MAX_RX_QUEUE_PER_LCORE =  (MAX_CORES / 2);
static constexpr uint32_t MAX_TX_QUEUE_PER_LCORE =  MAX_RX_QUEUE_PER_LCORE;

/* Others */
static constexpr uint32_t MAX_NB_STREAMS = 64;
static constexpr uint32_t DEF_NB_MBUF = 8192;
static constexpr uint32_t DRIVER_MIN_RX_PKTS = 4;
static constexpr uint32_t BUFFER_SPLIT_MP0 = 60;
static constexpr uint32_t BUFFER_SPLIT_NB_SEGS = 2;

static constexpr uint32_t MEM_DEVMEM = 1;
static constexpr uint32_t MEM_HOST_PINNED = 0;
static constexpr uint32_t GAP_PKTS = 4;
static constexpr uint32_t MAX_WTIME_NS = 20000000; // 20ms

static constexpr uint32_t GPU_PAGE_SHIFT = 16;
static constexpr uint32_t GPU_PAGE_SIZE = (1UL << GPU_PAGE_SHIFT);
static constexpr uint32_t GPU_PAGE_OFFSET = (GPU_PAGE_SIZE - 1);
static constexpr uint32_t GPU_PAGE_MASK = (~GPU_PAGE_OFFSET);

static constexpr uint32_t CPU_PAGE_SIZE = 4096;

static constexpr uint32_t NV_MIN_PIN_SIZE = 4;

enum workload_flags {
	NO_WORKLOAD       	= 0,
	CPU_WORKLOAD      	= 1 << 0,
	GPU_WORKLOAD      	= 1 << 1,
	GPU_PK_WORKLOAD   	= 1 << 2,
	GPU_GRAPHS_WORKLOAD = 1 << 3
};

///////////////////////////////////////////////////////////////////////////
//// Memory structs
///////////////////////////////////////////////////////////////////////////
struct burst_item {
	uint32_t status;
	uintptr_t addr[MAX_MBUFS_BURST];
	uint32_t len[MAX_MBUFS_BURST];

	struct rte_mbuf * mbufs[MAX_MBUFS_BURST];
	int num_mbufs;
	uint64_t bytes;
};

#endif
