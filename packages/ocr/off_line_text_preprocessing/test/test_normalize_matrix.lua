src_name       = arg[1]
line_mat       = matrix.fromFilename(arg[2])
top_prop       = tonumber(arg[3]) or 0.2
base_prop      = tonumber(arg[4]) or 0.1
dst_size       = tonumber(arg[5]) or 40

img = ImageIO.read(src_name):to_grayscale():invert_colors()

_,w,h = img:geometry()

printf("#Generating aspect ratio image\n")
dst_name = src_name .. "-aspect.png"
printf("%s->%s\n", src_name, dst_name)
img_dest = ocr.off_line_text_preprocessing.normalize_from_matrix(img, top_prop, base_prop,line_mat)
img_dest = img_dest:invert_colors()
ImageIO.write(img_dest, dst_name)
collectgarbage("collect")
