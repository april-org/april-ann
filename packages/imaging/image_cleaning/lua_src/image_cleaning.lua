image = image or {}
image.image_cleaning = image.image_cleaning or {}

-- Apply neural filter to an image
function image.image_cleaning.apply_filter_std(img, neighbors, clean_net)
  -- Generate the dataset
  local mImg = img:matrix()
  local tDim = mImg:dim()

  local side = neighbors*2 + 1

  local dsParams = {
     patternSize  = { side, side },
     offset       = { -neighbors, -neighbors },
     stepSize     = { 1,1 },
     numSteps     = tDim,
     defaultValue = 1,
     circular     = { false, false }
  }

  local dsInput = dataset.matrix(mImg, dsParams)

  -- Generate output dataset
  local mClean = matrix(tDim[1], tDim[2])
  local paramsClean = {
     patternSize  = {1, 1},
     stepSize     = {1, 1},
     numSteps     = { tDim[1], tDim[2] },
     defaultValue = 0,
  }

  local dsClean = dataset.matrix(mClean, paramsClean)

  --Apply the net to the dataset
  --
  clean_net:use_dataset {
      input_dataset  = dsInput,
      output_dataset = dsClean
  }

  local imgClean = Image(mClean)

  return imgClean
end

-- Apply neural filter to an image
function image.image_cleaning.apply_filter_histogram(img, neighbors, levels, radius, clean_net)
  local mImg = img:matrix()
  local tDim = mImg:dim()

  -- Compute image histogram
  local side  = neighbors*2 + 1
  local mHist = img:get_window_histogram(levels, radius)
  
  -- Generate the dataset
  local dsParams = {
     patternSize  = { side, side },
     offset       = { -neighbors, -neighbors },
     stepSize     = { 1,1 },
     numSteps     = tDim,
     defaultValue = 1,
     circular     = { false, false }
  }

  local dsHist = {
     patternSize  = {1, 1, levels},
     stepSize     = {1, 1, levels},
     numSteps     = { mHist:dim()[1], mHist:dim()[2],1 },
     defaultValue = 0,
     circular     = {false, false, false}
  }

  local dsDirty = dataset.matrix(mImg, dsParams)
  local dsHist  = dataset.matrix(mHist, dsHist)
  
  local dsInput = dataset.join{dsDirty, dsHist}

  -- Generate output dataset
  local mClean = matrix(tDim[1], tDim[2])
  local paramsClean = {
     patternSize  = {1, 1},
     stepSize     = {1, 1},
     numSteps     = { tDim[1], tDim[2] },
     defaultValue = 0,
  }

  local dsClean = dataset.matrix(mClean, paramsClean)
  --Apply the net to the dataset
  --
  clean_net:use_dataset {
      input_dataset  = dsInput,
      output_dataset = dsClean
  }

  local imgClean = Image(mClean)

  return imgClean
end


-- Returns a Dataset for iterate the image pixel by pixel
function image.image_cleaning.getSingleDataset(img, vecinos_salida)

    img_matrix = img:matrix()

    local lado_salida = (vecinos_salida*2+1)
    local params_limpia = {
        patternSize   = {lado_salida, lado_salida},
        offset        = {-vecinos_salida, -vecinos_salida},
        stepSize      = {1,1},
        numSteps      = img_matrix:dim(),
        defaultValue  = 1, -- 1 es blanco
        circular={false,false}
    }


    return dataset.matrix(img_matrix, params_limpia)
end


april_set_doc("image.image_cleaning.getCleanParameters",
	      {
		class="function",
		summary="Function to generete a sets of parameters given an image. ",
		description=
		  {
		    "This function takes an image and generates feature for each pixel",
		    "Some parameters include raw pixels values around the desired pixels",
		    "other have agreggated values like histogram or median filter.",
        "The given dataset is optimized in order to do not have copies of duplicated features."
		  },
		params= {
		  { "A image object."},
		  { "A table with parameters with the features to be extracted.",
      "All the features are optional and only will be generated the given features in the table",
        "- window = ... size of the side of a sliding window set to each pixel",
        "- histogram_levels = ... number of the values in the histogram, each the values is  computed in range 1/histogram_levels",
        "- histogram_radio = ... size of the side of a sliding window around the pixel where the histogram will be computed",
        "- median = ... size of the side of a sliding window around the pixel where the median value is computed",
        "- vertical = ... if defined generates features of the vertical column of each pixel.",
        "- horizontal = ... if defined generates features of the vertical column of each pixel.",
        "- random_perturbation = ... adds a gaussian perturbation of mean 0.5",
		  },
		},
		outputs= {
		  {"It returns a width*height length dataset with the features for each pixel."
		    },
		}
})
function image.image_cleaning.getCleanParameters(img, params)

  local function getWindowDataset(img, vecinos)
      local img_matrix = img:matrix()
      local lado = vecinos*2+1

      local params_sucia = {
          patternSize  = {lado,lado},
          offset       = {-vecinos,-vecinos},
          stepSize     = {1,1},
          numSteps     = img_matrix:dim(),
          defaultValue = 1, -- 1 es blanco
          circular     = {false,false}
      }

      return dataset.matrix(img_matrix, params_sucia)
  end

  local function getHistogramDataset(img, levels, radio)

      local mHist  = img:get_window_histogram(levels, radio)

      local params_sucia_hist = {
          patternSize  = {1,1, levels},
          stepSize     = {1,1, levels},
          numSteps     = {mHist:dim()[1], mHist:dim()[2], 1},
          defaultValue = 0,
          circular     = {false,false, false},
      }

      return dataset.matrix(mHist, params_sucia_hist)
  end

  local function getVerticalHistogram(img, levels, radius)
      local mVert = img:get_vertical_histogram(levels, radius)
      local params_vert_hist = {
          patternSize = {1, 8},
          stepSize = {0, 1, 0},
          numSteps = { img:matrix():dim()[1], img:matrix():dim()[2], 8},
          defaultValue = 0,
          circular = {false, false},
      }

      return dataset.matrix(mVert, params_vert_hist)
  end

  local function getHorizontalHistogram(img, levels, radius)
      local mHor = img:get_horizontal_histogram(levels, radius)
      local params_hor_hist = {
          patternSize = {1, 8},
          stepSize = {1,0,0},
          numSteps = { img:matrix():dim()[1],img:matrix():dim()[2], 8},
          defaultValue = 0,
          circular = {false, false},
      }

      return dataset.matrix(mHor, params_hor_hist)
  end

  local function getMedian(img, radio_mediana, vecinos)
      local img_median = img:median_filter(radio_mediana)

      local lado_mediana = 1 + vecinos*2
      params_median = {
          patternSize={lado_mediana,lado_mediana},
          offset={-vecinos,-vecinos},
          stepSize={1,1},
          numSteps=img_median:matrix():dim(),
          defaultValue = 1, -- 1 es blanco
          circular={false,false}
      }

      return dataset.matrix(img_median:matrix(), params_median)
  end
  local table_datasets = {}

  if params.window then
      local ds_window  = getWindowDataset(img, params.window)
      table.insert(table_datasets, ds_window)
  end

  if params.histogram_radio then
      local ds_hist    = getHistogramDataset(img, params.histogram_levels, params.histogram_radio)
      table.insert(table_datasets, ds_hist)
  end
  if params.median then
      local ds_median = getMedian(img, params.median, 0)
      table.insert(table_datasets, ds_median)
  end
  if params.horizontal then
      local ds_hor = getHorizontalHistogram(img, params.histogram_levels, params.horizontal)
      table.insert(table_datasets, ds_hor)
  end
  if params.vertical then
      local ds_ver = getVerticalHistogram(img, params.histogram_levels, params.vertical)
      table.insert(table_datasets, ds_ver)
  end

  local ds_sucia = dataset.join(table_datasets)
  -- 5. Añadir perturbación
  -- anyadimos un ruido gaussiano a la imagen sucia
  if params.random_perturbation then
      ds_sucia = dataset.perturbation{
          dataset   = ds_sucia,
          random    = params.random_perturbation or 123,
          mean      = 0,                     -- de la gaussiana
          variance  = params.variance_perturbation or 0.001, -- de la gaussiana
      }
  end
  return ds_sucia

end

function image.image_cleaning.getParametersFromString(param_str)
  
    params = {}
    
    -- window
    v = string.match(param_str, "_v(%d+)_")
    if v then
        params.window = tonumber(v)
    end
    levels = string.match(param_str, "_levels_(%d+)_")
    -- histogram_levels
    if levels then
        params.histogram_levels = tonumber(levels)
    end
    -- histogram_radio
    histogram = string.match(param_str, "_hist_(%d+)_")
    if histogram then
        params.histogram_radio = tonumber(histogram)
    end
    -- median
    median = string.match(param_str, "_median_(%d+)_")
    if median then
        params.median = tonumber(median)
    end
    -- vertical
    vertical = string.match(param_str, "_vertical_(%d+)_")
    if vertical then
        params.vertical = tonumber(vertical)
    end
    -- horizontal
    horizontal = string.match(param_str, "_horizontal_(%d+)_")
    if horizontal then
        params.horizontal = tonumber(horizontal)
    end
    --[[for i, v in pairs(params) do
        
        print(i,v)
    end
    ]]
    return params
end


function image.image_cleaning.clean_image(img, net, params) 

    -- Generates the parameters

    local dsInput = image.image_cleaning.getCleanParameters(img, params)

    local img_dims = img:matrix():dim()
    local mClean = matrix(img_dims[1], img_dims[2])
    local paramsClean = {
        patternSize  = {1, 1},
        stepSize     = {1, 1},
        numSteps     = { img_dims[1], img_dims[2] },
        defaultValue = 0,
    }

    local dsClean = dataset.matrix(mClean, paramsClean)


    -- Use the dataset
    net:use_dataset {
        input_dataset = dsInput,
        output_dataset = dsClean
    }
    -- Returns the image
    local imgClean = Image(mClean)

    return imgClean

end

