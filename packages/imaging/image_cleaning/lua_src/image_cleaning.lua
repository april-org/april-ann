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

  local function getVerticalHistogram(img, levels)
      local mVert = img:get_vertical_histogram(levels)
      local params_vert_hist = {
          patternSize = {1, 8},
          stepSize = {0, 1, 0},
          numSteps = { img:matrix():dim()[1], img:matrix():dim()[2], 8},
          defaultValue = 0,
          circular = {false, false},
      }

      return dataset.matrix(mVert, params_vert_hist)
  end

  local function getHorizontalHistogram(img, levels)
      local mHor = img:get_horizontal_histogram(levels)
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
      local ds_hor = getHorizontalHistogram(img, params.horizontal)
      table.insert(table_datasets, ds_hor)
  end
  if params.vertical then
      local ds_ver = getVerticalHistogram(img, params.vertical)
      table.insert(table_datasets, ds_ver)
  end

  local ds_sucia = dataset.join(table_datasets)
  -- 5. Añadir perturbación
  -- anyadimos un ruido gaussiano a la imagen sucia

  ds_sucia = dataset.perturbation{
      dataset   = ds_sucia,
      random    = params.random_perturbation or 123,
      mean      = 0,                     -- de la gaussiana
      variance  = params.variance_perturbation or 0.1, -- de la gaussiana
  }

  return ds_sucia

end
