result_images_filename=arg[1]
target_images_filename=arg[2]

result_images_ds = {}
for line in io.lines(result_images_filename) do
  local m = ImageIO.read(line):to_grayscale():info()
  table.insert(result_images_ds, dataset.matrix(m,{
						  stepSize    = {1, 1},
						  patternSize = {1, 1},
						  numSteps    = {m:dim()[1],
								 m:dim()[2]}
						}))
end

target_images_ds = {}
for line in io.lines(target_images_filename) do
  local m = ImageIO.read(line):to_grayscale():info()
  table.insert(target_images_ds, dataset.matrix(m,{
						  stepSize    = {1, 1},
						  patternSize = {1, 1},
						  numSteps    = {m:dim()[1],
								 m:dim()[2]}
						}))
end

fm, pr, rc = dataset.fmeasure{
  result_dataset = dataset.union(result_images_ds),
  target_dataset = dataset.union(target_images_ds),
}

print(fm, pr, rc)
