# XBFS
---
Software (tested)
-----
g++ 5.4.0, CUDA 9.1,10.0
Compilation flag: -O3

---
Hardware (tested)
------
K80, P6000, Titan Xp, V100 (tested)

---
Compile
-----

make

---
Execute
------
Type: "./bfs" it will show you what is needed.

Tips: It needs a CSR formated graph (beg file and csr file). 

You could use the code from "tuple_text_to_bin.binary_csr" folder to convert a edge list (text formated) graph into CSR files (binary), e.g., if you have a text file called "test.txt", it will convert it into "test.txt_beg_pos.bin" and "test.txt_csr.bin". You will need these two files to run XBFS.

---
Converter: edge tuples to CSR
----
- Compile: make
- To execute: type "./text_to_bin.bin", it will show you what is needed
- Basically, you could download a tuple list file from [snap](https://snap.stanford.edu/data/). Afterwards, you could use this converter to convert the edge list into CSR format. 

**For example**:

- Download https://snap.stanford.edu/data/com-Orkut.html file. **unzip** it. 
- **./text_to_bin.bin soc-orkut.mtx 1 2(could change, depends on the number of lines are not edges)**
- You will get *soc-orkut.mtx_beg_pos.bin* and *soc-orkut.mtx_csr.bin*. 
- You could use these two files to run enterprise.

---
Code specification
---------
The overall code structure of this project is:

- bfs_gpu_update.cu: main function.

- graph.hpp,graph.h: read the csr format graph file.

- swap.cuh: Function definition of swap module.

- bfs_adaptiveFQ_wb_async.cuh: runs XBFS.
- bfs_single_scan.cuh: BFS single scan codes on frontier queue generation.

 - bfs_TD_scan_free.cuh: BFS scan free codes on frontier queue generation.

 - prefix_sum1.cuh: It consists of prefix-sum modules used in XBFS.

- workload_gap.cuh: Module to find gap between the workloads.

- wtime.h: timing.

 


**Should you have any questions about this project, please contact us by hpda.lab@gmail.com.**

---
Reference
-------
[HPDC '19] XBFS: eXploring Runtime Optimizations for Breadth-First Search on GPUs 
PDF (coming soon)
Slides (coming soon)
Poster (coming soon)


[SC '15] Enterprise: Breadth-First Graph Traversal on GPUs [[PDF](http://home.gwu.edu/~asherliu/publication/enterprise_sc15.pdf)] [[Slides](http://home.gwu.edu/~asherliu/publication/enterprise_sc15.pdf)] [[Blog](http://home.gwu.edu/~asherliu/publication/enterprise_sc15.pdf)]

[SIGMOD '16] iBFS: Concurrent Breadth-First Search on GPUs [[PDF](http://home.gwu.edu/~asherliu/publication/ibfs.pdf)] [[Slides](http://home.gwu.edu/~asherliu/publication/ibfs_slides.pdf)] [[Poster](http://home.gwu.edu/~asherliu/publication/ibfs_poster.pdf)]
