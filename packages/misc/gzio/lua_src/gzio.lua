gzio_init()

-- modificamos el io.open
io.old_open = io.open
io.open = function(name, mode)
	    local f
	    if string.get_extension(name) == "gz" then
	      f = gzio.open(name, mode)
	    else f = io.old_open(name, mode)
	    end
	    return f
	  end
