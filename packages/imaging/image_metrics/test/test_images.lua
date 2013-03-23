if #arg ~= 2 then
    return
end

pred_img = ImageIO.read(arg[1]):to_grayscale()
gt_img = ImageIO.read(arg[2]):to_grayscale()

local params_ref = {
  patternSize = {1,1},
  offset      = {0,0},
  stepSize    = {1,1},
  numSteps    = pred_img:matrix():dim()

}

local params_gt = {
  patternSize = {1,1},
  offset      = {0,0},
  stepSize    = {1,1},
  numSteps    = gt_img:matrix():dim()

}
ds = dataset.matrix(pred_img:matrix(),params_ref)
gt = dataset.matrix(gt_img:matrix(),params_gt)



myMetrics = image.image_metrics()

myMetrics:process_dataset({
  predicted = ds,
  ground_truth = gt 
})

T = myMetrics:get_metrics()


printf("FMeasure: %f, PR: %d, RC: %f, MSE: %f, GA:%f, TNR: %f, ACC: %f, MPM: %f \n", T["FM"], T["RC"], T["PR"], T["MSE"],T["GA"], T["TNR"], T["ACC"], T["MPM"])


myMetricsBin = image.image_metrics()

myMetricsBin:process_dataset({
  predicted = ds,
  ground_truth = gt,
  binary = true,
  threshold = 0.5
})

T = myMetricsBin:get_metrics()

printf("FMeasure: %f, PR: %d, RC: %f, MSE: %f, GA:%f, TNR: %f, ACC: %f, MPM: %f \n", T["FM"], T["RC"], T["PR"], T["MSE"],T["GA"], T["TNR"], T["ACC"], T["MPM"])

local params = {"MSE", "MPM", "FM" }

image.image_metrics.printMetrics(myMetricsBin, "image1", params)
--myMetricsBin:printMetrics("image1", params)
