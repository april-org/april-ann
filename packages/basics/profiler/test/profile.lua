__profiled_program_name = arg[1]
table.remove(arg,1)
profiler.start()
dofile(__profiled_program_name)
profiler.stop()
profiler.save(io.open("profile.out", "w"))
