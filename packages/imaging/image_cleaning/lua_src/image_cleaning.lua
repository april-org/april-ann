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
