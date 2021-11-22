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

////////////////////////////////////////////////////////////////////////
//// Command line config params
////////////////////////////////////////////////////////////////////////
static uint32_t conf_enabled_port_mask = 0;
static int conf_mem_type = MEM_HOST_PINNED;
static int conf_gpu_id = 0;
static int conf_workload = CPU_WORKLOAD;
static int conf_port_id = 0;
static uint32_t conf_data_room_size = DEFAULT_MBUF_DATAROOM;
static uint32_t conf_pkt_burst_size = MAX_MBUFS_BURST;
static int conf_num_pipelines = 1;
static int conf_warmup_packets = 1000000;
static int conf_performance_packets = 10000000;
static int conf_nvprofiler = 0;
static int conf_mempool_cache = RTE_MEMPOOL_CACHE_MAX_SIZE;
static int conf_nb_mbufs = DEF_NB_MBUF;
static uint64_t conf_pktime_ns = 0;
static int conf_buffer_split = 0;

static const char short_options[] = 
	"b:"			/* burst size */
    "d:"			/* data room size */
    "g:"			/* GPU device id */
    "h"				/* help */
    "m:"			/* mempool type */
	"n"				/* Enable CUDA profiler */
	"p:"			/* num of pipelines */
	"s"				/* Enable buffer split */
	"t:"			/* Force execution time per packet */
    "v:"			/* performance packets */
	"w:"			/* workload type */
    "z:"			/* warmup packets */
    ;

static std::vector<Pipeline *> pipeline_v;

////////////////////////////////////////////////////////////////////////
//// GDRCopy
////////////////////////////////////////////////////////////////////////
gdr_t gdr_descr;
// Flush GPUDirect RDMA memory
gdr_mh_t flush_mh;
uintptr_t flush_d;
uintptr_t flush_h;
uintptr_t flush_free; // used to free devmem
size_t flush_size;

////////////////////////////////////////////////////////////////////////
//// DPDK config
////////////////////////////////////////////////////////////////////////
struct rte_ether_addr conf_ports_eth_addr[RTE_MAX_ETHPORTS];
struct rte_mempool *mpool_payload, *mpool_header;
struct rte_pktmbuf_extmem ext_mem;

static struct rte_eth_conf conf_eth_port = {
	.rxmode = {
				.mq_mode = ETH_MQ_RX_RSS,
				.max_rx_pkt_len = conf_data_room_size,
				.split_hdr_size = 0,
				.offloads = 0,
			},
	.txmode = {
			.mq_mode = ETH_MQ_TX_NONE,
			.offloads = 0,
			},
	.rx_adv_conf = {
			.rss_conf = {
						.rss_key = NULL,
						.rss_hf = ETH_RSS_IP
					},
			},
};

////////////////////////////////////////////////////////////////////////
//// Timers and statistics
////////////////////////////////////////////////////////////////////////
t_ns init_start;
t_ns init_end;
t_ns main_start;
t_ns main_end;

////////////////////////////////////////////////////////////////////////
//// Inter-threads communication
////////////////////////////////////////////////////////////////////////
volatile bool force_quit;

////////////////////////////////////////////////////////////////////////
//// Static functions
////////////////////////////////////////////////////////////////////////
static void print_stats(void)
{
	struct rte_eth_stats stats;
	int index_queue = 0;
	uint64_t tot_core_rx_pkts = 0, tot_core_rx_byte = 0, tot_core_rx_wbyte = 0;
	double max_core_rx_time = 0, avg_core_rx_pkts = 0, avg_core_rx_byte = 0, avg_core_rx_wbyte = 0;
	uint64_t tot_core_tx_pkts = 0, tot_core_tx_byte = 0;
	double max_core_tx_time = 0, avg_core_tx_pkts = 0, avg_core_tx_byte = 0;

	rte_eth_stats_get(conf_port_id, &stats);

	const char clr[] = { 27, '[', '2', 'J', '\0' };
	const char topLeft[] = { 27, '[', '1', ';', '1', 'H', '\0' };

	/* Clear screen and move to top left */
	printf("%s%s", clr, topLeft);

	printf("\n\nStats ===============================\n\n");
	printf("Warmup packets: %d, performance packets: %d\n", conf_warmup_packets, conf_performance_packets);
	printf("Initialization: %ld us (%ld sec)\n", Time::NsToUs(init_end-init_start).count(), Time::NsToSec(init_end-init_start).count());
	printf("Main loop:  %ld us (%ld sec)\n", Time::NsToUs(main_end-main_start).count(), Time::NsToSec(main_end-main_start).count() );

	printf("\n== RX:\n");

	printf("\tDPDK RX\n");
	for (index_queue = 0; index_queue < conf_num_pipelines; index_queue++)
		printf
		    ("\t\tQueue %d: packets = %ld bytes = %ld dropped = %ld\n",
		     index_queue, stats.q_ipackets[index_queue],
		     stats.q_ibytes[index_queue], stats.q_errors[index_queue]);

	printf("\tl2fwd-nv RX\n");
	for (index_queue = 0; index_queue < conf_num_pipelines; index_queue++) {
		printf("\t\tQueue %d: packets = %ld sec = %ld pkts/sec = %.2f\n", 
				index_queue,
				pipeline_v[index_queue]->rx_pkts, 
				Time::NsToSec(pipeline_v[index_queue]->stop_rx_core - pipeline_v[index_queue]->start_rx_core).count(),
				(float) (pipeline_v[index_queue]->rx_pkts / (float)(Time::NsToSec(pipeline_v[index_queue]->stop_rx_core - pipeline_v[index_queue]->start_rx_core).count()))
		);

		tot_core_rx_pkts += pipeline_v[index_queue]->rx_pkts;
	}

	printf("\n\tDPDK ipackets: %lu (%ld B), l2fwd-nv ipackets: %lu\n", stats.ipackets, stats.ibytes, tot_core_rx_pkts);

	printf("\n== TX:\n");

	printf("\tDPDK TX\n");
	for (index_queue = 0; index_queue < conf_num_pipelines; index_queue++)
		printf("\t\tQueue %d: packets = %ld bytes = %ld\n", index_queue, stats.q_opackets[index_queue], stats.q_obytes[index_queue]);

	printf("\tl2fwd-nv TX\n");
	for (index_queue = 0; index_queue < conf_num_pipelines; index_queue++) {
		printf("\t\tQueue %d: packets = %ld sec = %ld pkts/sec = %.2f\n", 
				index_queue,
				pipeline_v[index_queue]->tx_pkts, 
				Time::NsToSec(pipeline_v[index_queue]->stop_tx_core - pipeline_v[index_queue]->start_tx_core).count(),
				(float) (pipeline_v[index_queue]->tx_pkts / (float)(Time::NsToSec(pipeline_v[index_queue]->stop_tx_core - pipeline_v[index_queue]->start_tx_core).count()))
		);

		tot_core_tx_pkts += pipeline_v[index_queue]->tx_pkts;
	}

	printf("\n\tDPDK opackets: %lu (%ld B), l2fwd-nv opackets: %lu\n", stats.opackets, stats.obytes, tot_core_tx_pkts);

	printf("\n== ERRORS:\n");
	printf("Total of RX packets dropped by the HW, because there are no available buffer (i.e. RX queues are full)=%" PRIu64 "\n", stats.imissed);
	printf("Total number of erroneous received packets=%" PRIu64 "\n", stats.ierrors);
	printf("Total number of failed transmitted packets=%" PRIu64 "\n", stats.oerrors);
	printf("Total number of RX mbuf allocation failures=%" PRIu64 "\n", stats.rx_nombuf);
	printf("\n====================================================\n");
	fflush(stdout);
}

static int workload_with_gpu(int w) {
	if(w >= GPU_WORKLOAD)
		return 1;
	return 0;
}

////////////////////////////////////////////////////////////////////////
//// Cmd line options
////////////////////////////////////////////////////////////////////////
void print_opts(void)
{
	cudaError_t cuda_ret;
	struct cudaDeviceProp deviceProp;

	printf("============== INPUT OPTIONS ==============\n");
	printf("NV Mempool memory type = %s\n", (conf_mem_type == MEM_HOST_PINNED ? "Host pinned memory" : "Device memory"));
	printf("RX/TX queues = %d, Mempools x queue = 1\n", conf_num_pipelines);
	printf("Mbuf payload size = %d\n", conf_data_room_size);
	printf("mbufs per mempool = %d\n", conf_nb_mbufs);
	printf("Pipelines = %d (%d RX cores and %d TX cores)\n", conf_num_pipelines, conf_num_pipelines, conf_num_pipelines);

	if (conf_workload == NO_WORKLOAD)
		printf("Workload type = NO_WORKLOAD\n");
	else
	{
		if (conf_workload == CPU_WORKLOAD)
			printf("Workload type = CPU_WORKLOAD\n");
		if (conf_workload == GPU_WORKLOAD)
			printf("Workload type = GPU_WORKLOAD\n");
		if (conf_workload == GPU_PK_WORKLOAD)
			printf("Workload type = GPU_PK_WORKLOAD\n");
		if (conf_workload == GPU_GRAPHS_WORKLOAD)
			printf("Workload type = GPU_GRAPHS_WORKLOAD\n");
		if (workload_with_gpu(conf_workload) || conf_mem_type == MEM_DEVMEM)
		{
			CUDA_CHECK(cudaSetDevice(conf_gpu_id));
			cuda_ret = cudaGetDeviceProperties(&deviceProp, conf_gpu_id);
			if (cuda_ret != cudaSuccess)
				return;

			printf("Using GPU #%d, %s, %04x:%02x:%02x\n",
						conf_gpu_id, deviceProp.name,
						deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
		}

		printf("Per packet forced time = %lu ns\n", conf_pktime_ns);		
	}
	
	printf("Device port number = %d\n", conf_port_id);
	printf("RX pkts x burst = %d\n", conf_pkt_burst_size);
	if (conf_performance_packets > 0)
		printf("Warmup packets = %d, Performance packets = %d\n",
		       conf_warmup_packets, conf_performance_packets);
	else
		printf
		    ("Performance packets = infinite (stop execution with CTRL+C)\n");

	printf("NVTX profiler enabled = %s\n", (conf_nvprofiler == 1 ? "Yes" : "No"));
	printf("Buffer split enabled = %s\n", (conf_buffer_split == 1 ? "Yes" : "No"));

	printf("============================================\n\n");
}

static int parse_args(int argc, char **argv)
{
	int opt, ret = 0;
	char **argvopt;
	int option_index;
	char *prgname = argv[0];
	int totDevs;
	cudaError_t cuda_ret = cudaSuccess;
	argvopt = argv;

	while ((opt =
		getopt_long(argc, argvopt, short_options, NULL, &option_index)) != EOF) {

		switch (opt) {
			case 'b':
				conf_pkt_burst_size = l2fwd_parse_burst_size(optarg);
				if (conf_pkt_burst_size < 0 || conf_pkt_burst_size > MAX_MBUFS_BURST) {
					fprintf(stderr, "Invalid burst size (between 0 and %d)\n", MAX_MBUFS_BURST);
					l2fwdnv_usage(prgname);
					return -1;
				}
				break;

			case 'd':
				conf_data_room_size = l2fwd_parse_dataroom_size(optarg);
				if (conf_data_room_size < 0) {
					fprintf(stderr, "Invalid value for size of mbuf data room\n");
					l2fwdnv_usage(prgname);
					return -1;
				}
				break;

			case 'g':
				conf_gpu_id = l2fwd_parse_device_id(optarg);
				break;

			case 'h':
				l2fwdnv_usage(prgname);
				return -2;

			case 'm':
				conf_mem_type = l2fwd_parse_memtype(optarg);
				if (conf_mem_type < 0)
					return -1;
				break;

			case 'p':
				conf_num_pipelines = l2fwd_parse_pipelines(optarg);
				if (conf_num_pipelines <= 0) {
					fprintf(stderr, "Invalid value for number of pipelines %d\n", conf_num_pipelines);
					return -1;
				}
				break;
			
			case 's':
				conf_buffer_split = 1;
				break;

			case 'n':
				conf_nvprofiler = 1;
				break;

			case 't':
				conf_pktime_ns = l2fwd_parse_wtime(optarg);
				if ((conf_pktime_ns < 0) || (conf_pktime_ns > MAX_WTIME_NS)) {
					fprintf(stderr, "Workload packet time should be between 0 and %d ns (%lu provided)\n", MAX_WTIME_NS, conf_pktime_ns);
					l2fwdnv_usage(prgname);
					return -1;
				}

				break;

			case 'v':
				conf_performance_packets = l2fwd_parse_performance_packets(optarg);
				if (conf_performance_packets < 0) {
					fprintf(stderr, "invalid performance burst\n");
					l2fwdnv_usage(prgname);
					return -1;
				}

				break;

			case 'z':
				conf_warmup_packets = l2fwd_parse_warmup_packets(optarg);
				if (conf_warmup_packets < 0) {
					fprintf(stderr, "invalid warmup burst\n");
					l2fwdnv_usage(prgname);
					return -1;
				}

				break;

			case 'w':
				conf_workload = l2fwd_parse_workload(optarg);
				if (conf_workload < 0) {
					fprintf(stderr, "invalid conf_workload\n");
					return -1;
				}
				break;

				/* long options */
			case 0:
				break;

			default:
				l2fwdnv_usage(prgname);
				return -1;
		}
	}

	if (optind >= 0)
		argv[optind - 1] = prgname;

	ret = optind - 1;
	optind = 1;

	if (conf_mem_type == MEM_DEVMEM && conf_workload == CPU_WORKLOAD) {
		fprintf(stderr, "With device memory mempool, workload must be on GPU!\n");
		return -1;
	}

	if (conf_mem_type == MEM_DEVMEM || workload_with_gpu(conf_workload)) {
		cuda_ret = cudaGetDeviceCount(&totDevs);
		if (cuda_ret != cudaSuccess) {
			fprintf(stderr, "cudaGetDeviceCount error %d\n", cuda_ret);
			return -1;
		}

		if (totDevs < conf_gpu_id) {
			fprintf(stderr, "Erroneous GPU device ID (%d). Tot GPUs: %d\n", conf_gpu_id, totDevs);
			return -1;
		}
	}

	if ( ((conf_num_pipelines * 2) + 1) > (int)rte_lcore_count()) {
		fprintf(stderr,
			"Required conf_num_pipelines+1 (%d), cores launched=(%d)\n",
			conf_num_pipelines + 1, rte_lcore_count());
		return -1;
	}

	if (conf_num_pipelines > MAX_RX_QUEUE_PER_LCORE) {
		fprintf(stderr, "RX queues %d > MAX_RX_QUEUE_PER_LCORE %d\n", conf_num_pipelines, MAX_RX_QUEUE_PER_LCORE);
		return -1;
	}

	//Run l2fwd-nv until CTRL+C -- no time stats measuraments
	if (conf_performance_packets == 0)
		conf_warmup_packets = 0;

	if(conf_buffer_split && conf_workload == CPU_WORKLOAD) {
		fprintf(stderr, "Can't enable buffer split with CPU workload\n");
		return -1;
	}

	return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// RX CORE
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
static int rx_core(void *arg)
{
	long pipeline_idx = (long)arg;
	int nb_rx = 0, bindex = 0, nbytes = 0, nburst = 0, ngraph = 0;
	Pipeline * p_v = pipeline_v[pipeline_idx];
	struct burst_item * blist = p_v->burst_list;
	uint8_t flush_value = 0;
	uint32_t ret = 0;

	printf("Starting RX Core %u on queue %ld, socket %u\n", rte_lcore_id(), pipeline_idx, rte_socket_id());

	if (workload_with_gpu(p_v->workload_type))
		CUDA_CHECK(cudaSetDevice(conf_gpu_id));

	if(p_v->workload_type == GPU_GRAPHS_WORKLOAD)
	{
		CUDA_CHECK(cudaGraphLaunch(p_v->winstance[ngraph], p_v->c_stream));
		ngraph = (ngraph+1)%N_GRAPHS;
	}

	while (ACCESS_ONCE(force_quit) == 0 && ACCESS_ONCE(p_v->pipeline_force_quit) == 0)
	{
		if(ACCESS_ONCE(blist[bindex].status) != BURST_FREE)
		{
			fprintf(stderr, "Burst %d is not free. Pipeline it's too slow, quitting...\n", bindex);
			ACCESS_ONCE(force_quit) = 1;
			return -1;
		}

		PUSH_RANGE("rx_burst", 1);

		nb_rx = 0;
		while (
				ACCESS_ONCE(force_quit) == 0				&&
				ACCESS_ONCE(p_v->pipeline_force_quit) == 0	&&
				nb_rx < (conf_pkt_burst_size - GAP_PKTS)
		)
		{
			nb_rx += rte_eth_rx_burst(conf_port_id, pipeline_idx, 
										&(blist[bindex].mbufs[nb_rx]),
										(conf_pkt_burst_size - nb_rx));
		}

		POP_RANGE;

		if (!nb_rx)
			continue;

		p_v->rx_pkts += nb_rx;

		/* Activate RX timer after receiving first packets */
		if(p_v->start_rx_measure == false)
		{
			if(conf_performance_packets > 0 && p_v->rx_pkts >= conf_warmup_packets)
			{
				p_v->rx_pkts = 0;
				p_v->start_rx_measure = true;
				p_v->start_rx_core = Time::nowNs();
			}

			if(conf_performance_packets == 0)
			{
				p_v->start_rx_measure = true;
				p_v->start_rx_core = Time::nowNs();
			}
		}

		PUSH_RANGE("prep_pkts", 3);

		blist[bindex].num_mbufs = nb_rx;
		if (p_v->workload_type != NO_WORKLOAD)
		{
			for(int index=0; index < nb_rx; index++)
			{
				if(conf_buffer_split)
				{
					/*
					 * Must receive an mbufs with exactly BUFFER_SPLIT_NB_SEGS segments
					 */
					if(blist[bindex].mbufs[index]->nb_segs != BUFFER_SPLIT_NB_SEGS)
						rte_exit(EXIT_FAILURE, "Buffer split enabled but can't receive mbufs with %d segments\n", BUFFER_SPLIT_NB_SEGS);

					/* 
					 * Do MAC swap onto the CPU to simulate some CPU decisional work.
					 */
					struct rte_ether_hdr *eth = (struct rte_ether_hdr *) ((uint8_t *) (rte_pktmbuf_mtod_offset(blist[bindex].mbufs[index], void*, 0)));
					uint16_t * src_addr = (uint16_t *) (&eth->s_addr);
					uint16_t * dst_addr = (uint16_t *) (&eth->d_addr);
					uint16_t temp = dst_addr[0];
					dst_addr[0] = src_addr[0];
					src_addr[0] = temp;
					temp = dst_addr[1];
					dst_addr[1] = src_addr[1];
					src_addr[1] = temp;
					temp = dst_addr[2];
					dst_addr[2] = src_addr[2];
					src_addr[2] = temp;
#ifdef DEBUG_PRINT
					uint8_t *src = (uint8_t *) (&eth->s_addr);
					uint8_t *dst = (uint8_t *) (&eth->d_addr);
					fprintf
						(stderr, "#%d, mbuf_addr=%lx, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
							index, ((uint8_t *) (rte_pktmbuf_mtod_offset(blist[bindex].mbufs[index], void*, 0))), 
							src[0], src[1], src[2], src[3], src[4], src[5],
							dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]
						);
#endif
					/*
					 * Buffer provided to the GPU kernel is the second mbuf (segment) which resides in GPU memory
					 * GPU will still do the MAC swap of some bytes in the payload.
					 */
					blist[bindex].addr[index] 	= (uintptr_t) rte_pktmbuf_mtod_offset(blist[bindex].mbufs[index]->next, void*, 0);
					blist[bindex].len[index] 	= blist[bindex].mbufs[index]->next->data_len;
				}
				else
				{
					blist[bindex].addr[index] 	= (uintptr_t) rte_pktmbuf_mtod_offset(blist[bindex].mbufs[index], void*, 0);
					blist[bindex].len[index] 	= blist[bindex].mbufs[index]->data_len;
				}

				blist[bindex].bytes 		+= blist[bindex].mbufs[index]->pkt_len;
			}

			rte_wmb();
		}

		POP_RANGE;

		if (workload_with_gpu(p_v->workload_type))
			ACCESS_ONCE(blist[bindex].status) = BURST_READY;
		else
			ACCESS_ONCE(blist[bindex].status) = BURST_DONE;

		rte_wmb();

		if (p_v->workload_type == GPU_PK_WORKLOAD || p_v->workload_type == GPU_GRAPHS_WORKLOAD) {
			PUSH_RANGE("signal_pk", 4);

			ret = ACCESS_ONCE(((uint32_t*)flush_h)[0]);
			rte_mb();
			ACCESS_ONCE(((uint32_t*)(p_v->notify_kernel_list.ready_h))[bindex]) = BURST_READY;
			rte_mb();
			ret = ACCESS_ONCE(((uint32_t*)flush_h)[0]);

			POP_RANGE;

			if(p_v->workload_type == GPU_GRAPHS_WORKLOAD)
			{
				if(nburst == (GRAPH_BURST-1))
				{
					CUDA_CHECK(cudaGraphLaunch(p_v->winstance[ngraph], p_v->c_stream));
					ngraph = (ngraph+1)%N_GRAPHS;
					nburst = 0;
				}
				else
					nburst++;
			}
		} else if (p_v->workload_type == GPU_WORKLOAD) {
			PUSH_RANGE("macswap_gpu", 4);
			workload_launch_gpu_processing(
								blist[bindex].addr, blist[bindex].num_mbufs, &(blist[bindex].status), conf_pktime_ns,
								MAC_CUDA_BLOCKS, MAC_THREADS_BLOCK, p_v->c_stream
							);
			POP_RANGE;
		}

		if(p_v->start_rx_measure == true && conf_performance_packets > 0 && p_v->rx_pkts >= conf_performance_packets)
		{
			printf("Closing RX core (%ld), received packets = %ld\n", pipeline_idx, p_v->rx_pkts);
			fflush(stdout);
			ACCESS_ONCE(p_v->pipeline_force_quit) = 1;
			break;
		}

		bindex = (bindex+1) % MAX_BURSTS_X_QUEUE;
	}

	p_v->stop_rx_core = Time::nowNs();

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// TX CORE
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
static int tx_core(void *arg)
{
	long pipeline_idx = (long)arg;
	int nb_tx = 0, bindex = 0, nbytes = 0;
	Pipeline * p_v;
	p_v = pipeline_v[pipeline_idx];
	struct burst_item * blist = p_v->burst_list;
	uint8_t flush_value = 0;

	printf("Starting TX Core %u on queue %ld, socket %u\n", rte_lcore_id(), pipeline_idx, rte_socket_id());

	if (workload_with_gpu(p_v->workload_type))
		CUDA_CHECK(cudaSetDevice(conf_gpu_id));

	while(ACCESS_ONCE(force_quit) == 0 && ACCESS_ONCE(p_v->pipeline_force_quit) == 0)
	{
		PUSH_RANGE("wait_burst", 7);
		while(
				ACCESS_ONCE(force_quit) == 0 						&& 
				ACCESS_ONCE(p_v->pipeline_force_quit) == 0 	&&
				ACCESS_ONCE(blist[bindex].status) != BURST_DONE
			);
		rte_rmb(); //Avoid prediction
		POP_RANGE;

		p_v->tx_pkts += blist[bindex].num_mbufs;

		/* Activate RX timer after receiving first packets */
		if(p_v->start_tx_measure == false)
		{
			if(conf_performance_packets > 0 && p_v->tx_pkts >= conf_warmup_packets)
			{
				p_v->tx_pkts = 0;
				p_v->start_tx_measure = true;
				p_v->start_tx_core = Time::nowNs();
			}

			if(conf_performance_packets == 0)
			{
				p_v->start_tx_measure = true;
				p_v->start_tx_core = Time::nowNs();
			}
		}

		if (p_v->workload_type == CPU_WORKLOAD) {
			PUSH_RANGE("work_cpu", 7);
			workload_macswap_cpu(blist[bindex].addr, blist[bindex].num_mbufs, conf_pktime_ns);
			POP_RANGE;
		}

		PUSH_RANGE("rte_eth_tx_burst", 8);
		nb_tx = 0;
		while(
			ACCESS_ONCE(force_quit) == 0 				&&
			ACCESS_ONCE(p_v->pipeline_force_quit) == 0 	&&
			nb_tx < blist[bindex].num_mbufs
		) {
			nb_tx += rte_eth_tx_burst(conf_port_id, pipeline_idx,
										&(blist[bindex].mbufs[nb_tx]),
										blist[bindex].num_mbufs - nb_tx);
		}
		rte_wmb();
		POP_RANGE;

		if(p_v->start_tx_measure == true && conf_performance_packets > 0 && p_v->tx_pkts >= conf_performance_packets)
		{
			printf("Closing TX core (%ld), received packets = %ld\n", pipeline_idx, p_v->tx_pkts);
			fflush(stdout);
			break;
		}

		ACCESS_ONCE(blist[bindex].num_mbufs) 											= 0;
		ACCESS_ONCE(blist[bindex].bytes)												= 0;
		ACCESS_ONCE(blist[bindex].status)												= BURST_FREE;
		if (p_v->workload_type == GPU_PK_WORKLOAD || p_v->workload_type == GPU_GRAPHS_WORKLOAD)
			ACCESS_ONCE(((uint32_t*)(p_v->notify_kernel_list.ready_h))[bindex]) 	= BURST_FREE;
		rte_mb();

		bindex = (bindex+1) % MAX_BURSTS_X_QUEUE;
	}

	p_v->stop_tx_core = Time::nowNs();

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Signal Handler
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM || signum == SIGUSR1) {
		printf("\n\nSignal %d received, preparing to exit...\n", signum);
		ACCESS_ONCE(force_quit) = 1;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Main
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	struct rte_eth_dev_info dev_info;
	uint8_t socketid;
	int ret = 0, index_q, index_queue = 0, secondary_id = 0;
	uint16_t nb_ports;
	unsigned lcore_id;
	long icore = 0;
	uint16_t nb_rxd = RX_DESC_DEFAULT;
	uint16_t nb_txd = TX_DESC_DEFAULT;

	/* ====== Buffer split vars ====== */
	struct rte_eth_rxconf rxconf_qsplit;
	struct rte_eth_rxseg_split *rx_seg;
	union rte_eth_rxseg rx_useg[BUFFER_SPLIT_NB_SEGS] = {};
	/* =============================== */

	//Prevent any useless profiling
	cudaProfilerStop();

	printf("************ L2FWD-NV ************\n\n");

	init_start = Time::nowNs();

	/* ================ PARSE ARGS ================ */
	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");
	argc -= ret;
	argv += ret;

	/* parse application arguments (after the EAL ones) */
	ret = parse_args(argc, argv);
	if (ret == -2)
		rte_exit(EXIT_SUCCESS, "\n");
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Invalid NVL2FWD arguments\n");

	/* ================ FORCE QUIT HANDLER ================ */
	ACCESS_ONCE(force_quit) = 0;
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGUSR1, signal_handler);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Devices setup
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaSetDevice(conf_gpu_id);
	cudaFree(0);

	nb_ports = rte_eth_dev_count_avail();
	if (nb_ports == 0)
		rte_exit(EXIT_FAILURE, "No Ethernet ports - bye\n");

	rte_eth_dev_info_get(conf_port_id, &dev_info);
	printf("\nDevice driver name in use: %s... \n", dev_info.driver_name);

	if (strcmp(dev_info.driver_name, "mlx5_pci") != 0)
		rte_exit(EXIT_FAILURE, "Non-Mellanox NICs have not been validated in l2fwd-nv\n");
		// conf_eth_port.rx_adv_conf.rss_conf.rss_hf = ETH_RSS_IP;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// MEMPOOLS
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	ext_mem.elt_size = conf_data_room_size + RTE_PKTMBUF_HEADROOM;
	ext_mem.buf_len = RTE_ALIGN_CEIL(conf_nb_mbufs * ext_mem.elt_size, GPU_PAGE_SIZE);

	if (conf_mem_type == MEM_HOST_PINNED) {
		ext_mem.buf_ptr = rte_malloc("extmem", ext_mem.buf_len, 0);
		CUDA_CHECK(cudaHostRegister(ext_mem.buf_ptr, ext_mem.buf_len, cudaHostRegisterMapped));
		void *pDevice;
		CUDA_CHECK(cudaHostGetDevicePointer(&pDevice, ext_mem.buf_ptr, 0));
		if (pDevice != ext_mem.buf_ptr)
			rte_exit(EXIT_FAILURE, "GPU pointer does not match CPU pointer\n");
	} else {
		ext_mem.buf_iova = RTE_BAD_IOVA;
		CUDA_CHECK(cudaMalloc(&ext_mem.buf_ptr, ext_mem.buf_len));
		if (ext_mem.buf_ptr == NULL)
			rte_exit(EXIT_FAILURE, "Could not allocate GPU memory\n");

		unsigned int flag = 1;
		CUresult status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)ext_mem.buf_ptr);
		if (CUDA_SUCCESS != status) {
			rte_exit(EXIT_FAILURE, "Could not set SYNC MEMOP attribute for GPU memory at %llx\n", (CUdeviceptr)ext_mem.buf_ptr);
		}
		ret = rte_extmem_register(ext_mem.buf_ptr, ext_mem.buf_len, NULL, ext_mem.buf_iova, GPU_PAGE_SIZE);
		if (ret)
			rte_exit(EXIT_FAILURE, "Could not register GPU memory\n");
	}
	ret = rte_dev_dma_map(rte_eth_devices[conf_port_id].device, ext_mem.buf_ptr, ext_mem.buf_iova, ext_mem.buf_len);
	if (ret)
		rte_exit(EXIT_FAILURE, "Could not DMA map EXT memory\n");
	mpool_payload = rte_pktmbuf_pool_create_extbuf("payload_mpool", conf_nb_mbufs,
											0, 0, ext_mem.elt_size, 
											rte_socket_id(), &ext_mem, 1);
	if (mpool_payload == NULL)
		rte_exit(EXIT_FAILURE, "Could not create EXT memory mempool\n");

	if(conf_buffer_split)
	{
		mpool_header = rte_pktmbuf_pool_create("sysmem_mpool_hdr", conf_nb_mbufs,
												conf_mempool_cache, 0,
												BUFFER_SPLIT_MP0 + RTE_PKTMBUF_HEADROOM,
												rte_socket_id());
		if (!mpool_header) {
			rte_panic("Could not create sysmem mempool buffer split\n");
		}

		memcpy(&rxconf_qsplit, &dev_info.default_rxconf, sizeof(rxconf_qsplit));

		rxconf_qsplit.offloads = DEV_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
		rxconf_qsplit.rx_nseg = BUFFER_SPLIT_NB_SEGS;
		rxconf_qsplit.rx_seg = rx_useg;

		rx_seg = &rx_useg[0].split;
		rx_seg->mp = mpool_header;
		rx_seg->length = BUFFER_SPLIT_MP0;
		rx_seg->offset = 0;

		rx_seg = &rx_useg[1].split;
		rx_seg->mp = mpool_payload;
		rx_seg->length = 0;
		rx_seg->offset = 0;

		conf_eth_port.rxmode.offloads = DEV_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
		conf_eth_port.txmode.offloads = DEV_TX_OFFLOAD_MULTI_SEGS;
	}
	else
		conf_eth_port.rxmode.offloads = DEV_RX_OFFLOAD_JUMBO_FRAME;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// PORT 0 SETUP
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("Initializing port %u with %d RX queues and %d TX queues...\n", conf_port_id, conf_num_pipelines, conf_num_pipelines);
	ret = rte_eth_dev_configure(conf_port_id, conf_num_pipelines, conf_num_pipelines, &conf_eth_port);
	if (ret < 0)
		rte_exit(EXIT_FAILURE,
			 "Cannot configure device: err=%d, port=%u\n", ret,
			 conf_port_id);

	printf("Port RX offloads: ");
	print_rx_offloads(conf_eth_port.rxmode.offloads);
	printf("\n");
	printf("Port TX offloads: ");
	print_tx_offloads(conf_eth_port.txmode.offloads);
	printf("\n");

	ret = rte_eth_dev_adjust_nb_rx_tx_desc(conf_port_id, &nb_rxd, &nb_txd);
	if (ret < 0)
		rte_exit(EXIT_FAILURE,
			 "Cannot adjust number of descriptors: err=%d, port=%u\n",
			 ret, conf_port_id);

	rte_eth_macaddr_get(conf_port_id, &conf_ports_eth_addr[conf_port_id]);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// RX/TX QUEUES
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for (index_queue = 0; index_queue < conf_num_pipelines; index_queue++) {
		socketid = (uint8_t) rte_lcore_to_socket_id(index_queue);

		if(conf_buffer_split)
			ret = rte_eth_rx_queue_setup(conf_port_id, index_queue, nb_rxd, socketid, &rxconf_qsplit, NULL);
		else
			ret = rte_eth_rx_queue_setup(conf_port_id, index_queue, nb_rxd, socketid, NULL, mpool_payload);

		printf("\tQueue %d offloads: ", index_queue);
		print_rx_offloads(rxconf_qsplit.offloads);
		printf("\n");

		if (ret < 0)
			rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: err=%d, port=%u\n", ret, conf_port_id);
		
		ret = rte_eth_tx_queue_setup(conf_port_id, index_queue, nb_txd, socketid, NULL);
		if (ret < 0)
			rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup: err=%d, port=%u\n", ret, conf_port_id);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// START DEVICE
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	ret = rte_eth_dev_start(conf_port_id);
	if (ret != 0)
		rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n", ret, conf_port_id);

	rte_eth_promiscuous_enable(conf_port_id);
	printf("Port %d, MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n\n",
	       conf_port_id,
	       conf_ports_eth_addr[conf_port_id].addr_bytes[0],
	       conf_ports_eth_addr[conf_port_id].addr_bytes[1],
	       conf_ports_eth_addr[conf_port_id].addr_bytes[2],
	       conf_ports_eth_addr[conf_port_id].addr_bytes[3],
	       conf_ports_eth_addr[conf_port_id].addr_bytes[4],
	       conf_ports_eth_addr[conf_port_id].addr_bytes[5]);

	check_all_ports_link_status(conf_enabled_port_mask);

	print_opts();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// GDRCOPY FLUSH
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	gdr_descr = gdr_open();
	if (gdr_descr == NULL) {
		fprintf(stderr, "nv_init_gdrcopy failed\n");
		exit(EXIT_FAILURE);
	}

	if (0 != gdrcopy_alloc_pin(
								&(gdr_descr),
								&(flush_mh), &(flush_d), &(flush_h),
								&(flush_free), &(flush_size),
								sizeof(uint32_t))
	){
		fprintf(stderr, "gdrcopy_alloc_pin flush failed\n");
		exit(EXIT_FAILURE);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// BURST QUEUE PER PIPELINE
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	pipeline_v.reserve(conf_num_pipelines);
	for(int index_p = 0; index_p < conf_num_pipelines; index_p++)
		pipeline_v[index_p] = new Pipeline(index_p, conf_workload, conf_pktime_ns, &gdr_descr, index_p, index_p);

	if (conf_nvprofiler)
		cudaProfilerStart();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// START RX/TX CORES
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(icore = 0; icore < conf_num_pipelines; icore++)
	{
		PUSH_RANGE("launch_txcore", 5);
		secondary_id = rte_get_next_lcore(secondary_id, 1, 0);
		rte_eal_remote_launch(tx_core, (void *)icore, secondary_id);
		POP_RANGE;

		PUSH_RANGE("launch_rxcore", 6);
		secondary_id = rte_get_next_lcore(secondary_id, 1, 0);
		rte_eal_remote_launch(rx_core, (void *)icore, secondary_id);
		POP_RANGE;
	}

	init_end = Time::nowNs();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// WAIT RX/TX CORES
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	main_start = Time::nowNs();
	icore = 0;
	RTE_LCORE_FOREACH_SLAVE(icore) {
		if (rte_eal_wait_lcore(icore) < 0) {
			fprintf(stderr, "bad exit for coreid: %ld\n",
				icore);
			break;
		}
	}
	main_end = Time::nowNs();

	if (conf_nvprofiler)
		cudaProfilerStop();

	for(auto &p : pipeline_v)
		p->terminateWorkload();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Print stats
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	print_stats();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Close network device
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("Closing port %d...", conf_port_id);
	rte_eth_dev_stop(conf_port_id);
	rte_eth_dev_close(conf_port_id);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Final cleanup
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	pipeline_v.clear();
	gdr_close(gdr_descr);

	printf("Bye!\n");

	return 0;
}
