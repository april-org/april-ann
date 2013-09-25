FAULT TOLERANCE
---------------

- The MASTER could check if the workers could read DATA and execute script
  before run the task.

IMPROVEMENTS
------------

- It is better to use MPI messages instead of sockets.

- The definition of Lua classes which represent each of the possible messages,
  so this classes know how to be serialized and deserialized.

- A special message BUNCH, which could contain a SUMMARY of previous
  messages. It is not a collection of messages, it is a SUMMARY of a collection
  of messages, in the sense that a lot of network bandwith could be saved.

- The TASK could be run in MEMORY mode or DISK mode, if the map reduce result
  fits or not into memory, so the MASTER server will use a flag to define how to
  do the work.
