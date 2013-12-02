interest_points = interest_points or {}

local methods, class_metatable = class("interest_points.pointClassifier")

april_set_doc("interest_points.pointClassifier", {
		class       = "class",
		summary     = "Util for computing EyeFish transformation",
		description ={
		  "This class prepare the linear transformation.",
		}, })


april_set_doc("interest_points.pointClassifier.__Call", {
		class       = "method",
		summary     = "Constructor",
		description ={
		  "This class prepare the linear transformation.",
		},
    outputs = {"A Point Classifier object"},
})
function class_metatable:__call(ancho, alto, miniancho, minialto, reverse)

    -- prepara la configuracion para aplicar el ojo de pez con dataset.linear_comb
    -- local functions
    ------ convertir
    local function convertir(v,N)
        -- toma el vector y lo escala en N partes
        local total = 0
        local M = #v
        for i=1,M do total=total+v[i] end
        local resul = {}
        local pedazo = total/N
        local indice = 1
        local queda = pedazo
        for i=1,M do
            local trozo = v[i]
            while trozo > 0 and indice<=N do
                if trozo >= queda then
                    table.insert(resul,{i,indice,queda/v[i]})
                    trozo = trozo-queda
                    queda = pedazo
                    indice = indice+1
                else
                    table.insert(resul,{i,indice,trozo/v[i]})
                    queda = queda-trozo
                    trozo = 0
                end
            end
        end
        return resul
    end

    ------ crear_vector_estrecho
    local function crear_vector_estrecho(lado)
        local v = {}
        for i=1,lado do
            v[i] = 0.1+math.pow(1-math.pow(math.abs(i-0.5-(lado/2))/lado,2.5),300)
        end
        return v
    end

    ------ crear_vector_suave
    local function crear_vector_suave(lado)
        local v = {}
        for i=1,lado do
            v[i] = 0.1+math.pow(1-math.pow(math.abs(i-0.5-(lado/2))/lado,2),10)
        end
        return v
    end

    -- Funcion para invertir_tlc (Forwar tlc)
    local function invertir_tlc(tlc)
        
        local threshold = 0
        local inv_tlc = {}
        for i, lsources in pairs(tlc) do
            for j, v in ipairs(lsources) do
                local src, w = unpack(v)

                if inv_tlc[src] == nil then inv_tlc[src] = {} end
                if (w > threshold) then
                 table.insert(inv_tlc[src], {i-1, w})
                end
            end

        end


        return inv_tlc
    end
    -- function body
    local vancho = crear_vector_estrecho(ancho)
    local valto = crear_vector_suave(alto)

    local r_ancho = convertir(vancho,miniancho)
    local r_alto  = convertir(valto,minialto)
    local lenr_ancho = #r_ancho
    local lenr_alto  = #r_alto
    local tlc = {} -- tabla linear combination
    for i=1,lenr_alto do
        local y1,y,pesoy = unpack(r_alto[i])
        for j=1,lenr_ancho do
            local x1,x,pesox = unpack(r_ancho[j])
            local minipos = (y-1)*miniancho+x -- la posicion x,y en la matriz resultante
            if tlc[minipos] == nil then tlc[minipos] = {} end
            local pos = (y1-1)*ancho+x1
            table.insert(tlc[minipos],{pos,pesox*pesoy})
        end
    end
    -- normalizar cada cosa:
    for i,j in pairs(tlc) do
        local total = 0
        for k,l in ipairs(j) do
            total = total+l[2]
        end
        for k,l in ipairs(j) do
            l[2] = l[2]/total
        end
    end
    
    local obj = { }
    obj.ancho = ancho
    obj.alto = alto
    obj.miniancho = miniancho
    obj.minialto = minialto
    obj.table_inv = invertir_tlc(tlc)
    obj.inv_tlc = dataset.linear_comb_conf(obj.table_inv)
    obj.tlc = dataset.linear_comb_conf(tlc)
    obj.white = (reverse and 0) or 1
    
    obj = class_instance(obj, self, true)
    return obj

end

---------------------------------------------------
-- crop_image(img, x, y, tlc) -> table
---------------------------------------------------
function methods:crop_image(img, x, y)
    return self:applyLinearComb(img, x, y):getPattern(1)
end


april_set_doc("interest_points.pointClassifier.getFishDs", {
		class       = "method",
		summary     = "Main function for obtaining the FishEye Dataset",
		description ={
		  "Recieves an image and a point a generates the corresponding dataset",
		},
    params = {
      "An Image object",
      "X coordinate of the point",
      "Y coordinate of the point"
    },
    outputs = {"A Point Classifier object"},
})
function methods:getFishDs(img, x, y)

    local mFish = img:comb_lineal_forward(x,y, self.ancho, self.alto, self.miniancho, self.minialto, self.inv_tlc)
    local dsFish = dataset.matrix(mFish, {patternSize={self.miniancho*self.minialto},stepSize={self.miniancho*self.minialto}, numSteps={1} })

    return dsFish
end

function methods:applyLinearComb(img, x, y)
    local mat, _, _, dx, dy = img:matrix(),img:geometry()
    local ds = dataset.matrix(mat,
    {
        patternSize={self.alto,self.ancho},
        offset={(y+dy)-self.alto/2,(x+dx)-self.ancho/2},
        numSteps={1,1},
        defaultValue = self.white, -- pixel blanco
    })
    
    local dslc = dataset.linearcomb(ds,self.tlc)
    return dslc
end

-------------------------------------------------------------------------------
-- Given a point an mlp, return the output of the net with that point (window)
-----------------------------------------------------------------------------
april_set_doc("interest_points.pointClassifier.compute_point", {
		class       = "method",
		summary     = "Gets the net output for one point",
		description ={
      "Given a point and mlp, return the output of the net for that point",
		},
    params = {
      "An Image object",
      "X coordinate of the point",
      "Y coordinate of the point"
    },
    outputs = {"Table with the values of the softmax"},
})
function interest_points.pointClassifier:compute_point(img, x, y, mlp)

    local dsPoint = self:getFishDs(img, x, y)
    
    local dsOut = mlp:use_dataset{
      input_dataset = dsPoint 
    }
    return dsOut:getPattern(1)
end

april_set_doc("methods.getPointClass", {
		class       = "method",
		summary     = "Gets the corresponding class for the point",
		description ={
      "Given a point and mlp, return the max index for that point",
		},
    params = {
      "An Image object",
      "X coordinate of the point",
      "Y coordinate of the point"
    },
    outputs = {"Integer of the winning class"},
})
function methods:getPointClass(img, x, y, mlp)
    return self.compute_point(img, x, y, mlp):max()

end

--Gets a dataset and returns a table with the indexes of the major class
local function getIndexSoftmax(dsOut)
    local tResult = {}

    for i, v in dsOut:patterns() do
        _, p = table.max(v)
        table.insert(tResult, p)

    end
    return tResult
end
april_set_doc("interest_points.pointClassifier.compute_points", {
		class       = "method",
		summary     = "Return a dataset with the output of the net",
		description ={
      "Given a list of points and mlp, return the dataset corresponding to output of the mlp to that set of points",
		},
    params = {
      "An Image object",
      "List of points Tuples(x,y)",
      "Trainable Object"
    },
    outputs = {"Dataset of size Num_Classes x Num_Points"},
})
function methods:compute_points(img, points, mlp)

    --Compute the matrix (Forward)
    local fishEyes = {}

    for i, point in ipairs(points) do
        x,y = unpack(point)
        local dsFish = self:getFishDs(img, x, y)

        table.insert(fishEyes, dsFish)
    end

    local dsFishes = dataset.union(fishEyes)
    -- Classify the datasets
    local dsOut = mlp:use_dataset({input_dataset=dsFishes})

    return dsOut
end

----------------------------------------------------------
-- Given a image extract all the points and classify them
-- -------------------------------------------------------
function methods:classify_points(img, points, mlp)
   
    local dsOut = self:compute_points(img, points, mlp)
    local classes = getIndexSoftmax(dsOut)

    local res = {}
    for i, c in ipairs(classes) do
        x, y = unpack(points[i])
        table.insert(res,  {x,y,c, dsOut:getPattern(i)})
    end

    return res

    --[[  local scores = self:compute_points(img, points, mlp)
    local res = {}
    for i, score in ipairs(scores) do
    x = points[i][1]
    y = points[i][2]
    _, c = score:max()
    table.insert(res, {x, y, c})
    end
    return res
    --]]

end

function methods:extract_points(img, mlpUppers, mlpLowers)

    local uppers, lowers = interest_points.extract_points_from_image(img)

    -- Compute uppers
    local uppers_classified = self:classify_points(img, uppers, mlpUppers)
    -- Compute uppers
    local lowers_classified = pc:classify_points(img, lowers, mlpLowers)

    return uppers_classified, lowers_classified

end

----------------------------------------------------------------------------
-- table_points a (x, y, c) table where c is the majority class of the point
-- Utility function that gets a list of points and the class of the points
-- -------------------------------------------------------------------------
function interest_points.sort_by_class(table_points, classes)

    local res = {}
    for c = 1, classes do
      table.insert(res, {})
    end
    
    for _, point in ipairs(table_points) do
        x, y, c = unpack(point)
        table.insert(res[c], {x,y}) 
    end

    return res


end

-- Returns two tables wiht the lower and bound points classified
function interest_points.get_interest_points(img, mlpUppers, mlpLowers) 

    local uppers, lowers = interest_points.extract_points_from_image(img)
    pc = interest_points.pointClassifier(500, 250, 50, 30, false)

    -- Compute uppers
    local uppers_classified = pc:classify_points(img, uppers, mlpUppers)
    local classes = mlpUppers:get_output_size()
    local upper_table = interest_points.sort_by_class(uppers_classified, classes)

    -- Compute uppers
    local lowers_classified = pc:classify_points(img, lowers, mlpLowers)
    classes = mlpLowers:get_output_size()
    local lower_table = interest_points.sort_by_class(lowers_classified, classes)

    return upper_table, lower_table
end

----------------------------------------------------
-- Util functions
-- Recieve a table with the tag classes
-- Returns a Sparse (softmax) vector of the tags
-- ------------------------------------------------
function interest_points.loadTagDataset(points, numClasses)

    local numClasses = numClasses or 5
    local mOut = matrix(#points, points)
    local dsIndex = dataset.identity(numClasses)
    local dsOutIndex = dataset.matrix(mOut)
    local dsOut = dataset.indexed(dsOutIndex, {dsIndex})

    return dsOut
end
------------------------
--
-- Recieves an image and a interest point table, x_window and y_window parameters,
-- and return a returns a dataset of size num_points*((xwindow+1+xwindow*)*(y_window+1+ỳ_window))
--
--
---------------------------
function dataset.interest_point(img, table_points, x_window, y_window)

    img_matrix = img:matrix()

    -- Create the dataset over the image
    local params_img = {
      patternSize  = {x_window*2+1, y_window*2+1},
      offset       = {-x_window, -y_window},
      stepSize     = {1,1},
      numSteps     = img_matrix:dim(),
      defaultValue = 1,
      circular     = {false, false}
    }

    dsImg = dataset.matrix(img_matrix, params_img)

   print(dsImg:patternSize())

   -- Create the indexed point
   -- the table is composed by elems (x, y, c)
   tIndexes = table.imap(table_points, function (elem) return elem[1]*img_matrix:dim()[1]+elem[2] end)

   dsIndexes = dataset.matrix(matrix(tIndexes))

   dsPoints = dataset.indexed(dsIndexes, {dsImg})
    
   local dsOut = nil

   if #table_points[1] >= 3 then
      points = table.imap(table_points, function (elem) return elem[3] end)
   end
   

   return dsPoints, interest_points.loadTagDataset(points)
end
