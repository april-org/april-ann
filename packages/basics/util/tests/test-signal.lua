signal.register(signal.SIGQUIT, function() print("HOLA") end)
print(io.read("*l"))
signal.release(signal.SIGQUIT)
print(io.read("*l"))
