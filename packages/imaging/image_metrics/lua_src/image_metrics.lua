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

