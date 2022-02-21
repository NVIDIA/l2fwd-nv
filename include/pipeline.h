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

#ifndef PIPELINE_H
#define PIPELINE_H

#include "headers.h"
#include <chrono>
#include <time.h>

using t_ns = std::chrono::nanoseconds;
using t_us = std::chrono::microseconds;
using t_ms = std::chrono::milliseconds;
using t_s = std::chrono::seconds;
using t_tp = std::chrono::time_point<std::chrono::system_clock>;

class Time {
    public:
        Time() {}
        ~Time() {}

        static t_ns nowNs()
        {
            return std::chrono::system_clock::now().time_since_epoch();
        }

        static t_ns zeroNs()
        {
            return t_ns::zero();
        }

        static t_us NsToUs(t_ns time)
        {
            return std::chrono::duration_cast<t_us>(time);
        }

        static t_s NsToSec(t_ns time)
        {
            return std::chrono::duration_cast<t_s>(time);
        }

        static t_ns UsToNs(t_us time)
        {
            return std::chrono::duration_cast<t_ns>(time);
        }
};


class Pipeline {

	public:
		Pipeline(int _index, int _workload_type, int _pkt_time_ns, int _rx_queue, int _tx_queue, int _gpu_id);
		~Pipeline();
		void terminateWorkload();

		int index;
		int gpu_id;
		int workload_type;
		int pkt_time_ns;
		int rx_queue;
		int tx_queue;
		struct rte_gpu_comm_list *comm_list;
		cudaStream_t c_stream;
		// struct burst_item_sync * burst_sync_list;
		cudaGraph_t wgraph[N_GRAPHS];
		cudaGraphExec_t winstance[N_GRAPHS];
		int pipeline_force_quit;
		int rx_measure;
		int tx_measure;

		struct timespec temp_rx_start;
		struct timespec temp_rx_end;
		struct timespec temp_tx_start;
		struct timespec temp_tx_end;

		bool start_rx_measure;
		bool start_tx_measure;
		uint64_t rx_pkts;
		uint64_t tx_pkts;
		uint64_t rx_bytes;
		uint64_t tx_bytes;
		t_ns start_rx_core;
		t_ns stop_rx_core;
		t_ns start_tx_core;
		t_ns stop_tx_core;
};


#endif
