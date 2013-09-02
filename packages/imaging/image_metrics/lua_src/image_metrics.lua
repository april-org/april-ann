--ImageMetrics = ImageMetrics or {}
image  = image or {}
image.image_metrics = image.image_metrics or {}

function image.image_metrics.hello()

    print("Hola mundo")
end

-- Range indica el primer elemento de la columna, puede ser nil
function image.image_metrics.printMetrics(metrics, range, params)
   
    results = metrics:get_metrics()

    if range ~= nil then
       printf("%s\t",range)
    end
    
    for i,p in ipairs(params) do
        printf("%0.4f\t",results[p])
    end

    printf("\n")
end

function processImages(self, clean_img, gt_img)
  -- Load a dataset over the image and call process_dataset
    
    local dim_clean = clean_img:matrix():dim()  

    local clean_params = {
        patternSize   = {1, 1},
        offset        = {0,0},
        stepSize      = {1,1},
        numSteps      = dim_clean,
        defaultValue  = 1, -- 1 es blanco
        circular={false,false}
    }
    
    local dim_gt = gt_img:matrix():dim()  

    local gt_params   = {
        patternSize   = {1, 1},
        stepSize      = {1,1},
        numSteps      = dim_gt,
        defaultValue  = 1, -- 1 es blanco
        circular={false,false}
    }

    local ds_gt  = dataset.matrix(clean_img:matrix(), clean_params)
    local ds_clean = dataset.matrix(gt_img:matrix(), gt_params)

    self:process_dataset(
      { predicted = ds_clean,
        ground_truth = ds_gt,
      }
    )
end

class_extension(image.image_metrics, "processImages",processImages)

