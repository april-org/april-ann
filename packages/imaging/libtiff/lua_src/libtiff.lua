local libtiff_handlers = { read=libtiff.read, write=libtiff.write }
ImageIO.handlers["tif"] = libtiff_handlers
ImageIO.handlers["tiff"] = libtiff_handlers

