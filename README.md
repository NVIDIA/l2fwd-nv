# l2fwd-nv

In the vanilla l2fwd DPDK example each thread (namely, DPDK core) receives a burst (set) of packets,
does a swap of the src/dst MAC addresses and transmits back the same burst of modified packets.
l2fwd-nv is an improvement of l2fwd to show the usage of mbuf pool with GPU data buffers using
the vanilla DPDK API. The overall flow of the app is organised as follows:

* Create a number of pipelines, each one composed by:
    * one core to receive and accumulate bursts of packets (RX core) from a dedicated DPDK queue (RX queue)
    * a dedicated GPU/CPU workload entity which process the bursts
    * one core to transmit the burst of modified packets (TX core) using a dedicated DPDK queue (TX queue)
* For each pipeline, in a loop:
    * The RX core accumulates packets in bursts
    * The RX core triggers (asynchronously) the work (MAC swapping) on the received burst using CUDA kernel(s)
    * The TX core waits for the completion of the work
    * The TX core sends the burst of modified packets

Please note that a single mempool is used for all the DPDK RX/TX queues.
Using different command line option it's possible to:

* Create the mempool either in GPU memory or CPU pinned memory
* Decide how to do the MAC swapping in the packets:
    * No workload: MAC addresses are not swapped, l2fwd-nv is doing basic I/O forwarding
    * CPU workload: the CPU does the swap
    * GPU workload: a new CUDA kernel is triggered for each burst of accumulated packets
    * GPU persistent workload: a persistent CUDA kernel is triggered at the beginning on the CUDA stream dedicated to each pipeline. CPU has to communicate to this kernel that a new burst of packets has to be processed
    * GPU workload with CUDA graphs: a number of CUDA kernels is triggered for the next 8 bursts of packets
* Enable buffer split feature: each received packet is split in two mbufs. 60B into a CPU memory mbuf, remaning bytes are stored into the a GPU memory mbufs. The worklaod in this case is swapping some random bytes.

Please note that not all the combinations give the best performance.
This app should be considered a showcase to expose all the possibile combinations when dealing with GPUDirect RDMA and DPDK.
l2fwd-nv has a trivial workload that doesn't really require the use of CUDA kernels.

## Changelog

03/11/2022

* Updated to DPDK 22.03
* GDRCopy direct calls removed in favour of new `gpudev` cpu_map functions
* Code cleanup

11/26/2021

* Updated to the latest DPDK 21.11 release
* Introduced the new [gpudev library](https://github.com/DPDK/dpdk/blob/main/lib/gpudev/rte_gpudev.h)
* Benchmarks updated to latest MOFED 5.4, DPDK 21.11 and CUDA 11.4 with V100 and A100
* Benchmarks executed using testpmd as packet generator

## System configuration

Please note that DPDK 22.03 is included as submodule of this project and it's built locally with l2fwd-nv.

### Kernel configuration

Ensure that your kernel parameters include the following list:
```
default_hugepagesz=1G hugepagesz=1G hugepages=16 tsc=reliable clocksource=tsc intel_idle.max_cstate=0 mce=ignore_ce processor.max_cstate=0 audit=0 idle=poll isolcpus=2-21 nohz_full=2-21 rcu_nocbs=2-21 rcu_nocb_poll nosoftlockup iommu=off intel_iommu=off
```

Note that `2-21` corresponds to the list of CPUs you intend to use for the DPDK application and the value of this parameter needs to be changed depending on the HW configuration.

To permanently include these items in the kernel parameters, open `/etc/default/grub` with your favourite text editor and add them to the variable named `GRUB_CMDLINE_LINUX_DEFAULT`. Save this file, install new GRUB configuration and reboot the server:

```
$ sudo vim /etc/default/grub
$ sudo update-grub
$ sudo reboot
```

After reboot, verify that the changes have been applied.
As an example, to verify the system has 1 GB hugepages:

```
$ cat /proc/cmdline 
BOOT_IMAGE=/vmlinuz-5.4.0-53-lowlatency root=/dev/mapper/ubuntu--vg-ubuntu--lv ro maybe-ubiquity default_hugepagesz=1G hugepagesz=1G hugepages=16 tsc=reliable clocksource=tsc intel_idle.max_cstate=0 mce=ignore_ce processor.max_cstate=0 idle=poll isolcpus=2-21 nohz_full=2-21 rcu_nocbs=2-21 nosoftlockup iommu=off intel_iommu=off
$ grep -i huge /proc/meminfo
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
HugePages_Total:      16
HugePages_Free:       15
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:    1048576 kB
Hugetlb:        16777216 kB
```

### Mellanox network card

You need to follow few steps to configure your Mellanox network card.

* Download Mellanox OFED 5.4 from [here](http://www.mellanox.com/page/products_dyn?product_family=26)
* Enable CQE compression `mlxconfig -d <NIC PCIe address> set CQE_COMPRESSION=1`

If the Mellanox NIC supports IB and Ethernet mode (VPI adapters):
* Set the IB card as an Ethernet card `mlxconfig -d <NIC PCIe address> set LINK_TYPE_P1=2 LINK_TYPE_P2=2`
* Reboot the server or `mlxfwreset -d <NIC PCIe address> reset` and `/etc/init.d/openibd restart`

### NVIDIA GPU

Download and install the latest CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads).

### Install meson

DPDK 22.03 requires Meson > 0.49.2.

```
sudo apt-get install python3-setuptools ninja-build
wget https://github.com/mesonbuild/meson/releases/download/0.56.0/meson-0.56.0.tar.gz
tar xvfz meson-0.56.0.tar.gz
cd meson-0.56.0
sudo python3 setup.py install
```

### Enable GPUDirect RDMA

In order to enable GPUDirect RDMA with a Mellanox network card you need an additional kernel module.

If your system has CUDA 11.4 or newer, you need to load the `nvidia_peermem` module
that comes with the NVIDIA CUDA Toolkit.

```
sudo modprobe nvidia-peermem
```

More info [here](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#nvidia-peermem).

If your system has an older CUDA version you need to manually build and install the `nv_peer_memory` module.

```
git clone https://github.com/Mellanox/nv_peer_memory.git
cd nv_peer_memory
make
sudo insmod nv_peer_mem.ko
```

## Build the project

You can use cmake to build everything.

```
git clone --recurse-submodules https://github.com/NVIDIA/l2fwd-nv.git
cd l2fwd-nv
mkdir build
cd build
cmake ..
make -j$(nproc --all)
```

#### GDRCopy & gdrdrv

Starting from DPDK 22.03, GDRCopy has been embedded in DPDK and exposed through `rte_gpu_mem_cpu_map` function.
The CMakeLists.txt file automatically builds GDRCopy `libgdrapi.so` library.
After the build stage, you still need to launch `gdrdrv` kernel module on the system.

```
cd external/gdrcopy
sudo ./insmod.sh
```

Please note that, to enable GDRCopy in l2fwd-nv at runtime, you need to set the env var
`GDRCOPY_PATH_L` with the path to `libgdrapi.so` library which resides in
`/path/to/l2fwd-nv/external/gdrcopy/src`.

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


### Performance

In this section we report some performance analysis to highlight different l2fwd-nv configurations.
Benchmarks executed with between two different machines connected back-to-back,
one with l2fwd-nv and the other with testpmd.

We didn't observe any performance regression upgrading from DPDK 21.11 to DPDK 22.03.

#### l2fwd-nv machine

HW features:

* GIGABYTE E251-U70
* CPU Xeon Gold 6240R. 2.4GHz. 24C48T
* NIC ConnectX-6 Dx (MT4125 - MCX623106AE-CDAT)
* NVIDIA GPU V100-PCIE-32GB 
* NVIDIA GPU A100-PCIE-40GB 
* PCIe bridge between NIC and GPU: PLX Technology, Inc. PEX 8747 48-Lane, 5-Port PCI Express Gen 3 (8.0 GT/s)

HW topology between NIC and GPU:

```
-+-[0000:b2]-+-00.0-[b3-b6]----00.0-[b4-b6]--+-08.0-[b5]--+-00.0  Mellanox Technologies MT28841
 |           |                               |            \-00.1  Mellanox Technologies MT28841
 |           |                               \-10.0-[b6]----00.0  NVIDIA Corporation GV100GL [Tesla V100 PCIe 32GB]
 ```

SW features:

* Ubuntu 18.04 LTS
* Linux kernel 5.4.0-58-lowlatency
* GCC: 8.4.0 (Ubuntu 8.4.0-1ubuntu1~18.04)
* Mellanox OFED 5.4-3.1.0.0
* DPDK version: 21.11
* CUDA 11.4

Suggestes system configuration assuming a Mellanox network card with
bus id `b5:00.0` and network interface `enp181s0f0`:

```
mlxconfig -d b5:00.0 set CQE_COMPRESSION=1
mlxfwreset -d b5:00.0 r -y
ifconfig enp181s0f0 mtu 8192 up
ifconfig enp181s0f1 mtu 8192 up
ethtool -A enp181s0f0 rx off tx off
ethtool -A enp181s0f1 rx off tx off
sysctl -w vm.zone_reclaim_mode=0
sysctl -w vm.swappiness=0
```

PCIe Max Read Request:
```
$ sudo setpci -s b5:00.0 68.w
2930
$ setpci -s b5:00.0 68.w=5930
$ sudo lspci -s b5:00.0 -vvv | egrep "MaxRead"
		MaxPayload 256 bytes, MaxReadReq 4096 bytes
```

#### Packet generator

In the following performance report, we used the testpmd packet generator that comes with the DPDK 21.11 code.
The set of commands used to run and start testpmd is:

```
cd l2fwd-nv/external/dpdk/x86_64-native-linuxapp-gcc/app

sudo ./dpdk-testpmd -l 2-10 --main-lcore=2 -a b5:00.0 -- --port-numa-config=0,0 --socket-num=0 --burst=64 --txd=1024 --rxd=1024 --mbcache=512 --rxq=8 --txq=8 --forward-mode=txonly -i --nb-cores=8 --txonly-multi-flow

testpmd> set txpkts <pkt size>
start
```

#### Throughput measurement

In order to measure network throughput, we used the `mlnx_perf` application that comes with regular installation of MOFED.
Command line for mlnx_perf is:

```
mlnx_perf -i enp181s0f1
```

This tool reads network card numbers to determine number of sent and received bytes and calculate the data rate

```
tx_bytes_phy: 12,371,821,688 Bps = 98,974.57 Mbps
rx_bytes_phy: 12,165,283,124 Bps = 97,322.26 Mbps
```

#### I/O forwarding

In this test, GPU memory is used only to receive packets and transmit them back without and workload (I/O forwarding only)
in an infinite loop (no performance or warmup max packets). The number of packets received per workload (burst size `-b`)
is fixed to 64 packets.

Assuming a system with Mellanox network card bus id `b5:00.0` and an NVIDIA GPU with bus id `b6:00.0`, the command line used is:

```
sudo GDRCOPY_PATH_L=./external/gdrcopy/src ./build/l2fwdnv -l 0-9 -n 8 -a b5:00.1,txq_inline_max=0 -a b6:00.0 -- -m 1 -w 0 -b 64 -p 4 -v 0 -z 0
```

Please note that, if `libcuda.so` is not installed in the default system location, you need to specify
the path through the `CUDA_PATH_L=/path/to/libcuda.so` env var.

Network throughput measured with mlnx_perf:

| Packet bytes | Testpmd throughput | CPU memory throughput | GPU V100 memory throughput | GPU A100 memory throughput |
| ------------ | ------------------ | --------------------- | -------------------------- | -------------------------- |
| 64           | 74 Gbps            | 18 Gbps               | 19 Gbps                    | 19 Gbps                    |
| 128          | 82 Gbps            | 36 Gbps               | 37 Gbps                    | 37 Gbps                    |
| 256          | 82 Gbps            | 68 Gbps               | 67 Gbps                    | 67 Gbps                    |
| 512          | 97 Gbps            | 97 Gbps               | 94 Gbps                    | 95 Gbps                    |
| 1024         | 98 Gbps            | 98 Gbps               | 94 Gbps                    | 97 Gbps                    |

Please note that l2fwd-nv performance relies on the number of packets/sec rather than bytes/sec because the I/O
(and the workload) doesn't depend on the lenght of the packet.
In order to keep up with the line rate, in case of smaller packets,
the generator has to send more packets/sec than in case of 1kB packets.

#### Comparing GPU workloads

Here we compare I/O forwarding throughput using differnt GPU workloads:
* CUDA kernel (`-w 2`)
* CUDA persistent kernel (`-w 3`)
* CUDA Graph (`-w 4`)

Packet size is always 1kB, testpmd send throughput is ~98 Gbps and type of memory is GPU memory (`-m 1`).

Benchmarks with V100:

| Burst size  | CUDA kernel throughput | CUDA Persistent kernel throughput | CUDA Graphs throughput |
| ----------- | ---------------------- | --------------------------------- | ---------------------- |
| 16          | 18 Gbps                | 50 Gbps                           | 48 Gbps                |
| 32          | 37 Gbps                | 88 Gbps                           | 62 Gbps                |
| 64          | 90 Gbps                | 90 Gbps                           | 90 Gbps                |
| 128         | 90 Gbps                | 90 Gbps                           | 90 Gbps                |

Benchmarks with A100:

| Burst size  | CUDA kernel throughput | CUDA Persistent kernel throughput | CUDA Graphs throughput |
| ----------- | ---------------------- | --------------------------------- | ---------------------- |
| 16          | 23 Gbps                | 50 Gbps                           | 30 Gbps                |
| 32          | 49 Gbps                | 97 Gbps                           | 85 Gbps                |
| 64          | 97 Gbps                | 97 Gbps                           | 97 Gbps                |
| 128         | 97 Gbps                | 97 Gbps                           | 97 Gbps                |

## Caveats

### Packet size

If the packet generator is sending non-canonical packets sizes (e.g. 1514B) cache alignment problems may slow down the performance in case of GPU memory.
To enhance performance you may try to use the EAL param `rxq_pkt_pad_en=1` to the command line, e.g. `-w b5:00.1,txq_inline_max=0,rxq_pkt_pad_en=1`.

## References

More info in NVIDIA GTC'21 session `S31972 - Accelerate DPDK Packet Processing Using GPU` E. Agostini
