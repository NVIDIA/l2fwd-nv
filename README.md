# l2fwd-nv

In the vanilla l2fwd DPDK example each thread (namely, DPDK core) receives a burst of packets, does a swap of the src/dst MAC addresses and transmits back the same burst of modified packets.
l2fwd-nv is an improvement of l2fwd to show the usage of mbuf pool with GPU data buffers using the vanilla DPDK API. The overall flow of the app is organised as follows:

* Create a number of pipelines, each one composed by:
    * one core to receive and accumulate bursts of packets (RX core) from a dedicated DPDK queue (RX queue)
    * a dedicated GPU/CPU workload entity which process the bursts
    * one core to transmit the burst of modified packets (TX core) using a dedicated DPDK queue (TX queue)
* For each pipeline, in a loop:
    * The RX core accumulates packets in bursts
    * The RX core triggers (asynchronously) the work (MAC swapping) on the received burst using CUDA kernel
    * The TX core waits for the completion of the work
    * The TX core sends the burst of modified packets

Please note that a single mempool is used for all the DPDK RX/TX queues. Using different command line option it's possible to:

* Create the mempool either in GPU memory or CPU pinned memory
* Decide how to do the MAC swapping in the packets:
    * No workload: MAC addresses are not swapped, l2fwd-nv is doing basic I/O forwarding
    * CPU workload: the CPU does the swap
    * GPU workload: a new CUDA kernel is triggered for each burst of accumulated packets
    * GPU persistent workload: a persistent CUDA kernel is triggered at the beginning on the CUDA stream dedicated to each pipeline. CPU has to communicate to this kernel that a new burst of packets has to be processed
    * GPU workload with CUDA graphs: a number of CUDA kernels is triggered for the next 8 bursts of packets
* Enable buffer split feature: each received packet is split in two mbufs. 60B into a CPU memory mbuf, remaning bytes are stored into the a GPU memory mbufs. The worklaod in this case is swapping some random bytes.

Please note that not all the combinations give the best performance. This app should be considered a showcase to expose all the possibile combinations when dealing with GPUDirect RDMA and DPDK. l2fwd-nv has a trivial workload that doesn't really require the use of CUDA kernels.

## Build

### Install meson

DPDK 20.11 requires Meson > 0.47.1.
```
sudo apt-get install python3-setuptools ninja-build
wget https://github.com/mesonbuild/meson/releases/download/0.56.0/meson-0.56.0.tar.gz
tar xvfz meson-0.56.0.tar.gz
cd meson-0.56.0
sudo python3 setup.py install
```

### Build the project

You can use cmake to build everything.

```
cd l2fwd-nv
git submodule update --init
mkdir build
cd build
cmake ..
make -j$(nproc --all)
```

### Build and install GDRdrv

```
cd l2fwd-nv/subprojects/gdrcopy
make
sudo ./insmod.sh
```

## Benchmarks

l2fwd-nv with `-h` shows the usage and all the possible options

```
./build/l2fwdnv [EAL options] -- b|c|d|e|g|m|s|t|w|B|E|N|P|W
 -b BURST SIZE: how many pkts x burst to RX
 -d DATA ROOM SIZE: mbuf payload size
 -g GPU DEVICE: GPU device ID
 -m MEMP TYPE: allocate mbufs payloads in 0: host pinned memory, 1: GPU device memory
 -n CUDA PROFILER: Enable CUDA profiler with NVTX for nvvp
 -p PIPELINES: how many pipelines (each with 1 RX and 1 TX cores) to use
 -s BUFFER SPLIT: enable buffer split, 64B CPU, remaining bytes GPU
 -t PACKET TIME: force workload time (nanoseconds) per packet
 -v PERFORMANCE PKTS: packets to be received before closing the application. If 0, l2fwd-nv keeps running until the CTRL+C
 -w WORKLOAD TYPE: who is in charge to swap the MAC address, 0: No swap, 1: CPU, 2: GPU with one dedicated CUDA kernel for each burst of received packets, 3: GPU with a persistent CUDA kernel, 4: GPU with CUDA Graphs
 -z WARMUP PKTS: wait this amount of packets before starting to measure performance
```

To run l2fwd-nv in an infinite loop options `-z` and `-w` must be set to 0.
To simulate an hevier workload per packet, the `-t` parameter can be used to setup the number of nanoseconds per packet. This should help you to evaluate what's the best workload approach for your algorithm combining processing time per packet `-t` with number of packets per burst `-b`.

In the following benchmarks we report the forwarding throughput: assuming packet generator is able to transmit packets at the full linerate of 100Gbps, we're interested in the network throughput l2fwd-nv can reach retransmitting the packets.

#### Packet generator

In the following performance report, we used the T-Rex DPDK packet generator [v2.87](https://github.com/cisco-system-traffic-generator/trex-core/releases/tag/v2.87). You can reproduce the same benchmarks using testpmd (built in the DPDK submodule of this repo) using command line like the following one:

```
./subprojects/dpdk/x86_64-native-linuxapp-gcc/app/dpdk-testpmd -l 2-21 --main-lcore=2 -a b5:00.1 -- --port-numa-config=0,0 --socket-num=0 --burst=64 --txd=1024 --rxd=1024 --mbcache=512 --rxq=8 --txq=8 --forward-mode=txonly -i --nb-cores=8 --txonly-multi-flow

testpmd> set txpkts <pkt size>
start
```

### Performance

In this section we report some performance analysis to highlight different l2fwd-nv configurations. Benchmarks executed with between two different machines connected back-to-back, one with l2fwd-nv and the other with testpmd. l2fwd-nv machine HW topology between NIC and GPU:

```
-+-[0000:b2]-+-00.0-[b3-b6]----00.0-[b4-b6]--+-08.0-[b5]--+-00.0  Mellanox Technologies MT28841
 |           |                               |            \-00.1  Mellanox Technologies MT28841
 |           |                               \-10.0-[b6]----00.0  NVIDIA Corporation GV100GL [Tesla V100 PCIe 32GB]
 ```

l2fwd-nv machine HW features:
* GIGABYTE E251-U70
* CPU Xeon Gold 6240R. 2.4GHz. 24C48T
* NIC ConnectX-6 Dx (MT4125 - MCX623106AE-CDAT)
* NVIDIA GPU Tesla V100-PCIE-32GB 
* PCI bridge between NIC and GPU: PLX Technology, Inc. PEX 8747 48-Lane, 5-Port PCI Express Gen 3 (8.0 GT/s)

l2fwd-nv machine SW features:
* Ubuntu 18.04.5 LTS
* Linux kernel 5.4.0-53-lowlatency
* GCC: 8.4.0 (Ubuntu 8.4.0-1ubuntu1~18.04)
* Mellanox OFED: MLNX_OFED_LINUX-5.1-0.6.6.0
* DPDK version: 20.11
* CUDA 11.1

Suggestes system configuration:

```
mlxconfig -d <NIC Bus ID> set CQE_COMPRESSION=1
mlxfwreset -d <NIC Bus ID> r -y
ifconfig <NIC Interface name port 0> mtu 8192 up
ifconfig <NIC Interface name port 1> mtu 8192 up
ethtool -A <NIC Interface name port 0> rx off tx off
ethtool -A <NIC Interface name port 0> rx off tx off
setpci -s <NIC Bus ID> 68.w=5930
sysctl -w vm.zone_reclaim_mode=0
sysctl -w vm.swappiness=0
```

#### I/O forwarding

In this test, GPU memory is used only to receive packets and transmit them back without and workload (I/O forwarding only) in an infinite loop (no performance or warmup max packets).

```
./build/l2fwdnv -l 0-9 -n 8 -w b5:00.1,txq_inline_max=0 -- -m 1 -w 0 -b 64 -p 4 -v 0 -z 0
```
| Packet bytes | T-Rex throughput | CPU memory throughput | CPU memory packet loss | GPU memory throughput | GPU memory packet loss |
| ------------ | ---------------- | --------------------- | ---------------------- | --------------------- | ---------------------- |
| 64           | 100 Gbps         | 25 Gbps               | 74%                    | 24 Gbps               | 76%                    |
| 128          | 100 Gbps         | 43 Gbps               | 56%                    | 41 Gbps               | 58%                    |
| 256          | 95 Gbps          | 67 Gbps               | 30%                    | 69 Gbps               | 27%                    |
| 512          | 96 Gbps          | 95 Gbps               | 1%                     | 96 Gbps               | 1%                     |
| 1024         | 100 Gbps         | 99 Gbps               | 1%                     | 98 Gbps               | 1%                     |

Please note that l2fwd-nv performance relies on the number of packets/sec rather than bytes/sec because the I/O (and the workload) doesn't depend on the lenght of the packet. In order to keep up with the line rate, in case of smaller packets, the generator has to send more packets/sec than in case of 1kB packets.

#### Comparing GPU workloads

Here we compare I/O forwarding throughput using differnt GPU workloads: CUDA kernel (`-w 2`), CUDA persistent kernel (`-w 3`) and CUDA Graph (`-w 4`). Packet size is always 1kB and generator throughput is 100 Gbps and type of memory is GPU memory (`-m 1`).

| Burst size  | CUDA kernel throughput | CUDA kernel packet loss | CUDA Persistent kernel throughput | CUDA Persistent kernel packet loss | CUDA Graph throughput | CUDA Graph packet loss |
| ----------- | ---------------------- | ----------------------- | --------------------------------- | ---------------------------------- | -------------- | ---------------------- |
| 16          | 33 Gbps                 | 67%                    | 51 Gbps                           | 49%                                | 52 Gbps                 | 52%                    |
| 32          | 60 Gbps                 | 40%                    | 98 Gbps                           | 2%                                 | 98 Gbps                 | 2%                    |
| 64          | 98 Gbps                 | 2%                     | 98 Gbps                           | 2%                                 | 98 Gbps                 | 2%                    |
| 128         | 98 Gbps                 | 2%                     | 98 Gbps                           | 2%                                 | 98 Gbps                 | 2%                    |

## Caveats

### Packet size

If the packet generator is sending non-canonical packets sizes (e.g. 1514B) cache alignment problems may slow down the performance in case of GPU memory. To enhance performance you can use the EAL param `rxq_pkt_pad_en=1` to the command line, e.g. `-w b5:00.1,txq_inline_max=0,rxq_pkt_pad_en=1`.

## References

More info in NVIDIA GTC'21 session `S31972 - Accelerate DPDK Packet Processing Using GPU` E. Agostini
