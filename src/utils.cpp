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

#include "headers.h"

uint64_t get_timestamp_ns(void)
{
    struct timespec t;
    int             ret;
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if(ret != 0)
	{
    	fprintf(stderr, "clock_gettime failed\n");
		return 0;
    }
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

void wait_ns(uint64_t ns)
{
    uint64_t end_t = get_timestamp_ns() + ns, start_t = 0;
    while((start_t = get_timestamp_ns()) < end_t)
    {
        for(int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
        {
            __asm__ __volatile__("");
        }
    }
}

void wait_until_ns(uint64_t end_t)
{
    while(get_timestamp_ns() < end_t)
    {
        for(int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
        {
            __asm__ __volatile__("");
        }
    }
}

/* ================== WORKLOAD ================== */
void workload_macswap_cpu(struct rte_gpu_comm_pkt * pkt_list, int nmbuf, uint64_t wtime_ns)
{
	struct rte_ether_hdr *eth;
	uint8_t *data_ptr;
	uint16_t temp;
	uint16_t *src_addr, *dst_addr;
	int i = 0;
	uint64_t start;

	if (pkt_list == NULL || nmbuf <= 0)
		return;

	for (i = 0; i < nmbuf; i++) {

		if(wtime_ns)
			start = get_timestamp_ns();

		eth = (struct rte_ether_hdr *) (uint8_t *) pkt_list[i].addr;
		src_addr = (uint16_t *) (&eth->src_addr);
		dst_addr = (uint16_t *) (&eth->dst_addr);

#ifdef DEBUG_PRINT
		uint8_t *src = (uint8_t *) (&eth->src_addr);
		uint8_t *dst = (uint8_t *) (&eth->dst_addr);
		printf
		    ("#%d, mbuf_addr=%lx, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
		     i, addr[i], src[0], src[1], src[2], src[3], src[4], src[5],
		     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]
		    );
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

		if(wtime_ns)
			wait_until_ns(start+wtime_ns);
			// fprintf(stderr, "Wait until start %lu time_ns %lu end %lu\n", start, wtime_ns, start+wtime_ns);
	}
}
/* =================================================== */

/* ================== CMD LINE OPTIONS ================== */
void l2fwdnv_usage(const char *prgname)
{
	printf("\n\n%s [EAL options] -- b|c|d|e|g|m|s|t|w|B|E|N|P|W\n"
	       " -b BURST SIZE: how many pkts x burst to RX\n"
	       " -d DATA ROOM SIZE: mbuf payload size\n"
	       " -g GPU DEVICE: GPU device ID\n"
	       " -m MEMP TYPE: allocate mbufs payloads in 0: host pinned memory, 1: GPU device memory\n"
		   " -n CUDA PROFILER: Enable CUDA profiler with NVTX for nvvp\n"
		   " -p PIPELINES: how many pipelines (each with 1 RX and 1 TX cores) to use\n"
		   " -s BUFFER SPLIT: enable buffer split, 64B CPU, remaining bytes GPU\n"
		   " -t PACKET TIME: force exec time (nanoseconds) per packet\n"
	       " -v PERFORMANCE PKTS: packets to be received before closing the application. If 0, l2fwd-nv keeps running until the CTRL+C\n"
		   " -w WORKLOAD TYPE: who is in charge to swap the MAC address, 0: No swap, 1: CPU, 2: GPU with one dedicated CUDA kernel for each burst of received packets, 3: GPU with a persistent CUDA kernel, 4: GPU with CUDA Graphs\n"
	       " -z WARMUP PKTS: wait this amount of packets before starting to measure performance\n",
	       prgname);
}

unsigned int l2fwd_parse_device_id(const char *q_arg)
{
	char *end = NULL;
	unsigned long n;

	n = strtoul(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return 0;

	return n;
}

uint32_t l2fwd_parse_dataroom_size(const char *q_arg)
{
	char *end = NULL;
	uint32_t n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;
	if (n <= 0 || n > DEFAULT_MBUF_DATAROOM)
		return -1;
	return n;
}

uint32_t l2fwd_parse_burst_size(const char *q_arg)
{
	char *end = NULL;
	uint32_t n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;
	if (n > DEFAULT_PKT_BURST_INPUT)
		return -1;
	return n;
}

int l2fwd_parse_pipelines(const char *q_arg)
{
	char *end = NULL;
	int n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;
	if (n < 0)
		return -1;
	if (n > MAX_PIPELINES)
		return -1;

	return n;
}

int l2fwd_parse_warmup_packets(const char *q_arg)
{
	char *end = NULL;
	int n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;

	if (n > 0 && n < DRIVER_MIN_RX_PKTS)
		return DRIVER_MIN_RX_PKTS;

	return n;
}

int l2fwd_parse_performance_packets(const char *q_arg)
{
	char *end = NULL;
	int n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;

	if (n > 0 && n < DRIVER_MIN_RX_PKTS)
		return DRIVER_MIN_RX_PKTS;

	return n;
}

int l2fwd_parse_workload(const char *q_arg)
{
	char *end = NULL;
	int n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;

	if (n == 1)
		return CPU_WORKLOAD;
	if (n == 2)
		return GPU_WORKLOAD;
	if (n == 3)
		return GPU_PK_WORKLOAD;
	if (n == 4)
		return GPU_GRAPHS_WORKLOAD;

	return NO_WORKLOAD;
}

int l2fwd_parse_memtype(const char *q_arg)
{
	char *end = NULL;
	int n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;

	if (n == MEM_DEVMEM)
		return MEM_DEVMEM;

	return MEM_HOST_PINNED;
}

int l2fwd_parse_wtime(const char *q_arg)
{
	char *end = NULL;
	int n;

	n = strtol(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;
	if (n > MAX_WTIME_NS)
		return -1;
	if (n == 0) {
		return 1;
	}
	return n;
}


/* Check the link status of all ports in up to 9s, and print them finally */
void check_all_ports_link_status(uint32_t port_mask)
{
#define CHECK_INTERVAL 100	/* 100ms */
#define MAX_CHECK_TIME 90	/* 9s (90 * 100ms) in total */
	uint16_t portid;
	uint8_t count, all_ports_up, print_flag = 0;
	struct rte_eth_link link;

	//printf("\nChecking link status");
	for (count = 0; count <= MAX_CHECK_TIME; count++) {
		//if (force_quit) return;
		all_ports_up = 1;
		RTE_ETH_FOREACH_DEV(portid) {
			//if (force_quit) return;
			if ((port_mask & (1 << portid)) == 0)
				continue;
			memset(&link, 0, sizeof(link));
			rte_eth_link_get_nowait(portid, &link);
			/* print link status if flag set */
			if (print_flag == 1) {
				if (link.link_status)
					printf("Port%d Link Up. Speed %u Mbps - %s\n",
						portid, link.link_speed,
						(link.link_duplex == RTE_ETH_LINK_FULL_DUPLEX) ? ("full-duplex") : ("half-duplex\n"));
				else
					printf("Port %d Link Down\n", portid);
				continue;
			}
			/* clear all_ports_up flag if any link down */
			if (link.link_status == RTE_ETH_LINK_DOWN) {
				all_ports_up = 0;
				break;
			}
		}
		/* after finally printing all link status, get out */
		if (print_flag == 1)
			break;

		if (all_ports_up == 0) {
			//printf(".");
			fflush(stdout);
			rte_delay_ms(CHECK_INTERVAL);
		}

		/* set the print_flag if all ports up or timeout */
		if (all_ports_up == 1 || count == (MAX_CHECK_TIME - 1)) {
			print_flag = 1;
			//printf("done\n");
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Print Offloads
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void print_rx_offloads(uint64_t offloads)
{
        uint64_t single_offload;
        int begin;
        int end;
        int bit;

        if (offloads == 0)
                return;

        begin = __builtin_ctzll(offloads);
        end = sizeof(offloads) * CHAR_BIT - __builtin_clzll(offloads);

        single_offload = 1ULL << begin;
        for (bit = begin; bit < end; bit++) {
                if (offloads & single_offload)
                        printf(" %s",
                               rte_eth_dev_rx_offload_name(single_offload));
                single_offload <<= 1;
        }
}

void print_tx_offloads(uint64_t offloads)
{
	uint64_t single_offload;
	int begin;
	int end;
	int bit;

	if (offloads == 0)
		return;

	begin = __builtin_ctzll(offloads);
	end = sizeof(offloads) * CHAR_BIT - __builtin_clzll(offloads);

	single_offload = 1ULL << begin;
	for (bit = begin; bit < end; bit++) {
		if (offloads & single_offload)
			printf(" %s",
			       rte_eth_dev_tx_offload_name(single_offload));
		single_offload <<= 1;
	}
}