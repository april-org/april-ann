
ref = matrix.fromString[[
10 1
ascii
1
1
1
1
1
1
1
0
0
0]]


pred = matrix.fromString[[
10 1
ascii
0
0
1
1
1
1
1
0
0
0]]

ds = dataset.matrix(pred)
gt = dataset.matrix(ref)
myMetrics = image.image_metrics()

myMetrics:process_dataset({
    predicted = ds,
    ground_truth = gt 
})

res = myMetrics:get_metrics()

printf("FMeasure: %f, PR: %f, RC: %f, RC, GA:%f,MSE: %f, TNR: %f, ACC: %f\n",res.FM, res.PR, res.RC, res.GA, res.MSE, res.TNR, res.ACC)

