# Iterative Map Reduce

Based on paper: *Iterative MapReduce for Large Scale Machine
Learning*. J. Rosen, N. Polyzotis, V. Borkar, Y. Bu, M.J. Carey, M. Weimer,
T. Condie, R. Ramakrishnan. *arXiv.org*.

There are four main scripts:

- `MapReduceMaster.lua`: which runs the master of the MapReduce operation.

- `MapReduceWorker.lua`: which runs a worker on any host machine.

- `MapReduceTest.lua`: useful for do testing of your MapReduce jobs, before to
run it on the cluster. It uses the same running scheme, but using sequential
operations, instead of grid computing. Note that this script must be executed
with only a tiny part of the data.

- `MapReduceTask.lua`: executes the task on the client machine, asking the
master for do MapReduce operations, and processing the result on the client
machine (showing at screen, or saving at disk, or whatever you want).

## MapReduceMaster

## MapReduceWorker

## MapReduceTest

## MapReduceTask
