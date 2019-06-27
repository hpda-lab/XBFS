exe = bfs 

cc = "$(shell which nvcc)" 
#flags = -I. -fopenmp -march=athlon64 -O3
flags = -I. -O3
#flags += -std=c++11

ifeq ($(debug), 1)
	flags+= -DDEBUG 
endif

objs = $(patsubst %.cu,%.o,$(wildcard *.cu))

deps = $(wildcard ./*.cuh) \
		$(wildcard *.h) \
		Makefile

%.o:%.cu $(deps)
	$(cc) -c $< -o $@ $(flags)

$(exe):$(objs)
	$(cc) $(objs) -o $(exe) $(flags)

test:$(exe)
	./$(exe) LiveJournal/LJ_beg_pos.bin LiveJournal/LJ_csr.bin LiveJournal/LJ_weight.bin 128 128 1 LiveJournal/LJ.csv 0.1 1 LJ 1

clean:
	rm -rf $(exe) $(objs) 
