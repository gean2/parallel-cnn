# SKip list implementations and benchmarking code

The skip list implementation can be found in the include/ directory. We have:
* skiplist.h: Superclass / interface for skip list implementations
* synclist.hpp: Implementation of a coarse-grained locking skip list
* finelock.hpp: Implementation of Herlihy's fine-grained locking skip list
* lockfree.hpp: Implementation of Fraser's lock-free skip list

The other code files are used for benchmarking and analysis. We have:
* include/utils.h + utils.cpp: utils file for testing and benchmarking, e.g. key generation
* benchmark.cpp / ghc_benchmark.cpp: benchmarking files to time executions with different parameters; used to get measurements for benchmarks and graphs
* analysis.cpp: runs a specified distribution of operations on a desired skip list type; used to get measurements for perf report