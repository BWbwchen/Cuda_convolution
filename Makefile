NVFLAGS  := -std=c++11 -G -g -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm -lpng -lz
EXES     := final

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

final: final.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
