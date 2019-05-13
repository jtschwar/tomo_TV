debug_or_optimize = -O2

CXX = g++
CXXFLAGS = -Wall -Werror -Wno-div-by-zero -pedantic --std=c++11 $(debug_or_optimize)  -I ./Dependencies/Eigen/ 

main: main.cpp tlib.hpp tlib.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@

debug: debug_or_optimize=-g

# disable built-in rules
.SUFFIXES:

# these targets do not create any files
.PHONY: clean
clean :
	rm -vrf *.o *.exe *.gch *.dSYM *.out calc List_compile_check
