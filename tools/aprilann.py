import numpy

class matrix(object):
    @staticmethod
    def fromMMap(filename):
	sz = 4
	f = open(filename)
	magic_number = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=1)
	if magic_number != 0x4321:
	    raise RuntimeError("Incorrect magic number")
	commit_number = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=1)
	size = numpy.fromfile(f,dtype=numpy.dtype('uint64'),count=1)
	buf = numpy.fromfile(f,dtype=numpy.dtype('float32'),count=size/sz)
	version = numpy.fromfile(f,dtype=numpy.dtype('uint32'),count=1)
	if version != 0x00000001:
	    raise RuntimeError("Incorrect matrix version number")
	num_dim = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=1)
	stride = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=num_dim)*sz
	offset = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=1)*sz
	matrix_size = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=num_dim)
	total_size = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=1)
	last_raw_pos = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=1)
	major_order = numpy.fromfile(f,dtype=numpy.dtype('int32'),count=1)
	transposed = numpy.fromfile(f,dtype=numpy.dtype('int8'),count=1)
	return numpy.ndarray(shape = tuple(matrix_size),
			     strides = tuple(stride),
			     dtype = numpy.dtype('float32'),
			     buffer = buf,
			     offset = offset,
			     order = 'C' if (major_order == 0) else 'F')
