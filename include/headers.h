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

#ifndef L2FWDNV_H
#define L2FWDNV_H

///////////////////////////////////////////////////////////////////////////
//// Generic headers
///////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <netinet/in.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <vector>

///////////////////////////////////////////////////////////////////////////
//// DPDK headers
///////////////////////////////////////////////////////////////////////////
#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_metrics.h>
#include <rte_bitrate.h>
#include <rte_latencystats.h>
#include <rte_gpudev.h>

///////////////////////////////////////////////////////////////////////////
//// Local headers
///////////////////////////////////////////////////////////////////////////
#include "cuda_related.h"
#include "common.h"
#include "pipeline.h"

///////////////////////////////////////////////////////////////////////////
//// Memory structs
///////////////////////////////////////////////////////////////////////////
struct port_statistics {
	uint64_t tx_pkts[MAX_TX_QUEUE_PER_LCORE];
	uint64_t tx_brst[MAX_TX_QUEUE_PER_LCORE];
	uint64_t tx_bytes[MAX_TX_QUEUE_PER_LCORE];
	uint64_t tx_tot_ns[MAX_TX_QUEUE_PER_LCORE];

	uint64_t rx_pkts[MAX_RX_QUEUE_PER_LCORE];
	uint64_t rx_brst[MAX_RX_QUEUE_PER_LCORE];
	uint64_t rx_bytes[MAX_RX_QUEUE_PER_LCORE];
	uint64_t rx_warmup_bytes[MAX_RX_QUEUE_PER_LCORE];
	uint64_t rx_tot_ns[MAX_RX_QUEUE_PER_LCORE];

	uint64_t dropped[MAX_RX_QUEUE_PER_LCORE];
} __rte_cache_aligned;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Function declarations
/////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
// Workload
/////////////////////////////////////////////////////////////////
void workload_macswap_cpu(struct rte_gpu_comm_pkt * pkt_list, int nmbuf, uint64_t wtime_ns);

void workload_launch_gpu_processing(struct rte_gpu_comm_list * comm_list, uint64_t wtime_n,
		int cuda_blocks, int cuda_threads, cudaStream_t stream);

void workload_launch_persistent_gpu_processing(struct rte_gpu_comm_list * comm_item_list, uint64_t wtime_n,
		int cuda_blocks, int cuda_threads, cudaStream_t stream);

void workload_launch_gpu_graph_processing(struct rte_gpu_comm_list * comm_item, uint64_t wtime_n,
		int cuda_blocks, int cuda_threads, cudaStream_t stream);

/////////////////////////////////////////////////////////////////
// Command Line Options
/////////////////////////////////////////////////////////////////
void print_opts(void);
void l2fwdnv_usage(const char *prgname);
uint32_t l2fwd_parse_dataroom_size(const char *q_arg);
uint32_t l2fwd_parse_burst_size(const char *q_arg);
int l2fwd_parse_pipelines(const char *q_arg);
int l2fwd_parse_warmup_packets(const char *q_arg);
int l2fwd_parse_performance_packets(const char *q_arg);
int l2fwd_parse_workload(const char *q_arg);
int l2fwd_parse_memtype(const char *q_arg);
unsigned int l2fwd_parse_device_id(const char *q_arg);
int l2fwd_parse_mac_update(const char *q_arg);
int l2fwd_parse_wtime(const char *q_arg);
void check_all_ports_link_status(uint32_t port_mask);
void print_rx_offloads(uint64_t offloads);
void print_tx_offloads(uint64_t offloads);

/////////////////////////////////////////////////////////////////
// Time
/////////////////////////////////////////////////////////////////
uint64_t get_timestamp_ns(void);
void wait_ns(uint64_t ns);
void wait_until_ns(uint64_t end_t);

#endif