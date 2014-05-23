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
function class_metatable:__call(ancho, alto, miniancho, minialto, reverse_old)

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
    obj.white = (reverse_old and 1) or 1
    obj.reverse_old = reverse_old
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
    
    if self.reverse_old then
        mFish:complement()
    end
    local m = mFish:rewrap(self.minialto,self.miniancho)
   -- ImageIO.write(Image(m),"kk.png")
   -- ImageIO.write(img,"kk2.png")
    local dsFish = dataset.matrix(mFish, 
    {
        patternSize={self.miniancho*self.minialto},
        stepSize={self.miniancho*self.minialto},
        numSteps={1},
    
    })
        
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
function methods:compute_point(img, x, y, mlp)

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
function interest_points.getIndexSoftmax(dsOut)
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
    local classes = interest_points.getIndexSoftmax(dsOut)
    local res = {}
    for i, c in ipairs(classes) do
        
        x, y = unpack(points[i])
        table.insert(res,  {x,y,c, dsOut:getPattern(i)[c]})
        -- Get the confidence
    end
    

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
    return res

end

function methods:extract_points(img, mlpUppers, mlpLowers)

    local uppers, lowers
    local img_orig = img

    if (self.reverse_old)  then
      img = img:invert_colors()
    end

    -- The extraction is done by reversing the image
    uppers, lowers = interest_points.extract_points_from_image(img)

    img = img_orig
    -- Compute uppers
    local uppers_classified = self:classify_points(img, uppers, mlpUppers)
    -- Compute uppers
    local lowers_classified = self:classify_points(img, lowers, mlpLowers)

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

function interest_points.filter_points(points, h, w)

    local ASCENDER=1
    local UPPER_BASELINE=2
    local BODY=3
    local LOWER_BASELINE=4
    local DESCENDER=5

    local function filtrar_repetidos(mark_index)
        for x,tblx in pairs(mark_index) do
            for _, class in pairs({ASCENDER, UPPER_BASELINE}) do
                if mark_index[x][class] ~= nil and #mark_index[x][class] >= 2 then
                    local ymin = mark_index[x][class][1]
                    for idx, y in ipairs(mark_index[x][class]) do
                        if y < ymin then ymin = y end
                    end

                    local tbl_body = table.ifilter(mark_index[x][class], 
                    function(k,v) return v ~= ymin end)
                    mark_index[x][class] = {ymin}
                    mark_index[x][BODY] = table.join(mark_index[x][BODY], tbl_body)
                    --print("x=",x,"class=",class,"ymin=",ymin,"tbl:", table.concat(mark_index[x][class]),
                    -- "body: ",table.concat(tbl_body, ","))--,"body_old:", table.concat(mark_index[x][BODY], ","))
                end
            end

            for _, class in pairs({DESCENDER, LOWER_BASELINE}) do
                if mark_index[x][class] ~= nil and #mark_index[x][class] >= 2 then
                    --print("#mark_index[x][class]=", #mark_index[x][class])
                    local ymax = mark_index[x][class][1]
                    for idx, y in ipairs(mark_index[x][class]) do
                        if y > ymax then ymax = y end
                    end

                    local tbl_body = table.ifilter(mark_index[x][class], 
                    function(k,v) return v ~= ymax end)
                    mark_index[x][class] = {ymax}
                    mark_index[x][BODY] = table.join(mark_index[x][BODY], tbl_body)
                    --print("x=",x,"class=",class,"ymax=",ymax,"tbl:", table.concat(mark_index[x][class]),
                    --                "body: ",table.concat(tbl_body, ","))--,"body_old:", table.concat(mark_index[x][BODY], ","))
                end
            end
        end
    end

    -------------------------------------------
    -- filtrar_repetidos_array(tbl, func) -> table
    -- Elimina elementos con la x repetida de una tabla ordenada de puntos, dejando el que cumpla func (tipicamente min o max)
    -------------------------------------------
    local function filtrar_repetidos_array(tbl, func)
        local result = {}
        local y
        for i=1,#tbl do
            if y == nil then
                y = tbl[i][2]
            end
            if tbl[i+1]~= nil and tbl[i+1][1] == tbl[i][1] then
                y = func(y, tbl[i+1][2])
            else
                table.insert(result,{tbl[i][1], y})
                y=nil
            end
        end
        return result
    end

    ---------------------------
    -- argmax(tbl) -> value
    -- Devuelve el x tal que tbl[x] = max(tbl[i] para todo i en tbl)
    ---------------------------
    local function argmax(tbl)
        local wmax = 1
        local max  = tbl[1]
        for i,j in ipairs(tbl) do
            if j > max then
                wmax = i
                max  = j
            end
        end
        if max ~= nil then
            return wmax,max
        else 
            return nil, nil
        end
    end

    -------------------------------------------------------------
    -- get_first_point(tbl, width, default_y) -> point, index
    -------------------------------------------------------------
    local function get_first_point(tbl, width, default_y)
        if #tbl == 0 then
            return {0, default_y}, 0
        else
            local result = {0, tbl[1][2]}
            if tbl[1][1] == 0 then
                return result, 1
            else
                return result, 0
            end
        end
    end

    ---------------------------------------------------------------
    -- get_next_point(tbl, index, width, default_y) -> point
    ---------------------------------------------------------------
    local function get_next_point(tbl, index, width, default_y)
        if index+1 < 1 then error("invalid index") end

        if #tbl > 0 then
            if index < #tbl then
                return tbl[index+1]
            else
                return {width-1, tbl[#tbl][2]}
            end
        else
            return {width-1, default_y}
        end
    end
    local body = {}
    -------------------------------------------------------------
    -- get_first_point(tbl, width, default_y) -> point, index
    -------------------------------------------------------------
    local function get_first_point(tbl, width, default_y)
        if #tbl == 0 then
            return {0, default_y}, 0
        else
            local result = {0, tbl[1][2]}
            if tbl[1][1] == 0 then
                return result, 1
            else
                return result, 0
            end
        end
    end

    ---------------------------------------------------------------
    -- get_next_point(tbl, index, width, default_y) -> point
    ---------------------------------------------------------------
    local function get_next_point(tbl, index, width, default_y)
        if index+1 < 1 then error("invalid index") end

        if #tbl > 0 then
            if index < #tbl then
                return tbl[index+1]
            else
                return {width-1, tbl[#tbl][2]}
            end
        else
            return {width-1, default_y}
        end
    end

    local function filter_repeated(tbl, func)
        print ("Filtering")
        local result = {}
        local y
        for i=1,#tbl do
            if y == nil then
                y = tbl[i][2]
            end
            if i+1 <= #tbl and tbl[i+1][1] == tbl[i][1] then
                y = func(y, tbl[i+1][2])
            else
                table.insert(result,{tbl[i][1], y})
                y=nil
            end
        end
        return result
    end

    -- Mark index is genereted given the classe
    local mark_index = {}

    for i, t in ipairs(points) do
        for _, p in ipairs(t) do
            x, y = unpack(p)
            if not mark_index[x] then
                mark_index[x] = {}
            end
            if not mark_index[x][i] then
                mark_index[x][i] = {}
            end
            table.insert(mark_index[x][i],y)
        end
    end

    filtrar_repetidos(mark_index)
    -- START

    local tables = {{},{},{},{},{}}

    for x, t in pairs(mark_index) do
      for class, y_tbl in pairs(t) do
        for _,y in ipairs(y_tbl) do
          table.insert(tables[class], {x,y})
        end
      end
    end
    for class=1,5 do
      table.sort(tables[class], function(a,b) return a[1]<b[1] end)
    end
    local asc_idx, desc_idx, upper_idx
    local prev_asc, next_asc
    local prev_desc, next_desc

    prev_asc, asc_idx = get_first_point(tables[ASCENDER], w, 0)
    next_asc = get_next_point(tables[ASCENDER], asc_idx, w, 0)
    asc_idx = asc_idx+1

    local highest = {-1, math.huge}
    local new_ascenders = {}
    for i, p in ipairs(tables[BODY]) do
        --[[
        printf ("prev_asc = (%d, %d), p = (%d, %d), next_asc = (%d, %d)\n",
        prev_asc[1], prev_asc[2], p[1], p[2], next_asc[1], next_asc[2])
        --]]
        if p[1] > next_asc[1] or i == #tables[BODY] then
            local x,y = unpack(highest)
            local asc_y = prev_asc[2] + ((x-prev_asc[1])/(next_asc[1]-prev_asc[1]))*(next_asc[2]-prev_asc[2])
            --printf("---> highest=(%d,%d), asc_y=%d\n", x,y, asc_y)
            if y < asc_y then
                table.insert(new_ascenders, highest)
                --printf("inserted ascender: (%d, %d)\n", x, y)
            end
            while p[1] > next_asc[1] do
                prev_asc = next_asc
                next_asc = get_next_point(tables[ASCENDER], asc_idx, w, 0)
                asc_idx = asc_idx+1
            end
            highest = p 
        end
        if p[2] < highest[2] then
            highest=p
        end
    end

    tables[ASCENDER] = table.join(tables[ASCENDER], new_ascenders)
    table.sort(tables[ASCENDER], function(a,b) return a[1]<b[1] end)
    tables[ASCENDER] = filtrar_repetidos_array(tables[ASCENDER], math.min)

    prev_desc, desc_idx = get_first_point(tables[DESCENDER], w, h-1)
    next_desc = get_next_point(tables[DESCENDER], desc_idx, w, h-1)
    desc_idx = desc_idx+1

    local lowest = {-1, 0}
    local new_descenders = {}
    for i, p in ipairs(tables[BODY]) do
        --[[
        printf ("prev_desc = (%d, %d), p = (%d, %d), next_desc = (%d, %d)\n",
        prev_desc[1], prev_desc[2], p[1], p[2], next_desc[1], next_desc[2])
        --]]
        if p[1] > next_desc[1] or i == #tables[BODY] then
            local x,y = unpack(lowest)
            local desc_y = prev_desc[2] + ((x-prev_desc[1])/(next_desc[1]-prev_desc[1]))*(next_desc[2]-prev_desc[2])
            --printf("---> lowest=(%d,%d), desc_y=%d\n", x,y, desc_y)
            if y > desc_y then
                table.insert(new_descenders, lowest)
                --printf("inserted descender: (%d, %d)\n", x, y)
            end
            while p[1] > next_desc[1] do
                prev_desc = next_desc
                next_desc = get_next_point(tables[DESCENDER], desc_idx, w, h-1)
                desc_idx = desc_idx + 1
            end
            lowest = p
        end
        if p[2] > lowest[2] then
            lowest=p
        end
    end

    tables[DESCENDER] = table.join(tables[DESCENDER], new_descenders)
    table.sort(tables[DESCENDER], function(a,b) return a[1]<b[1] end)
    tables[DESCENDER] = filtrar_repetidos_array(tables[DESCENDER], math.max)


    return tables

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

-- Generate the tables for text line normalization
function methods:get_interest_points_line(img, mlpUppers, mlpLowers, threshold)

  local w, h = img:geometry()
  -- extract points and classify the points
  print ("Extracting points")
  local uppers_classified, lowers_classified = self:extract_points(img, mlpUppers, mlpLowers)
  print ("Well we have ", #uppers_classified, #lowers_classified)
  print (#uppers_classified[1])

  -- filter by the points by threshold and minima for uppers and maxima for lowers
  uppers_classified =  interest_points.filter_by_class_threshold(uppers_classified, threshold, true)
  lowers_classified = interest_points.filter_by_class_threshold(lowers_classified, threshold, false)
 
  print ("After the first filter we have ", #uppers_classified, #lowers_classified)
  
  -- Mix the tables
  print("lets mix all together")
  local tables = interest_points.classified_to_tables(5, uppers_classified
  , lowers_classified)
  for i, t in ipairs(tables) do
      print(i,#t) 
  end
  -- filter
  print("jorge is filtering...")
  tables = interest_points.filter_points(tables, height,width)
  for i, t in ipairs(tables) do
      print(i,#t) 
  end
  return tables
end

-- Util function
-- Recieve a list of points
-- the type of extrema (upper, lower)
-- return the list of points filtered by this criteria 
function interest_points.filter_by_class_threshold(points, threshold, upper)

    local result = {}

    for _, point in ipairs(points) do

        local class = point[3]
        if math.exp(point[4]) > threshold and
            ((upper and class < 3) or (class >3)) then
            table.insert(result, point)
        end
    end
    return result
end

-- Given a list of points (x,y,c,confidence)
-- returns 5 tables
function interest_points.classified_to_tables(num_classes, ...)
    -- TODO add confidence threshold!
    local result = {}
    for i = 1, num_classes do
        table.insert(result, {})
    end

    points = {...}
    for i, t in ipairs(points) do
        for j, v in ipairs(t) do
            c = v[3]
            table.insert(result[c],{v[1],v[2]})
        end

    end
    for i, t in ipairs(result) do  
        table.sort(t, function(a,b) return a[1] < b[1] end)
    end
    return result
end
----------------------------------------------------
-- Util functions
-- Recieve a table with the tag classes
-- Returns a Sparse (softmax) vector of the tags
-- ------------------------------------------------
function interest_points.loadTagDataset(points, numClasses)

    local numClasses = numClasses or 5
    local mOut = matrix(#points, points)

    local dsOut
    if numClasses == 2 then
        dsOut = dataset.matrix(mOut)
    else
        local dsIndex = dataset.identity(numClasses)
        local dsOutIndex = dataset.matrix(mOut)
        dsOut = dataset.indexed(dsOutIndex, {dsIndex})
    end
    return dsOut
end
------------------------
--
-- Recieves an image and a interest point table, x_window and y_window parameters,
-- and return a returns a dataset of size num_points*((xwindow+1+xwindow*)*(y_window+1+ỳ_window))
--
--
---------------------------
function dataset.interest_point(img, table_points, x_window, y_window, num_classes, reverse)

    local img_matrix = img:matrix()

    local white  = 1

    if reverse then
        white = 0
    end
    -- Create the dataset over the image
    local params_img = {
        patternSize  = {y_window*2+1, x_window*2+1},
        offset       = {-y_window, -x_window},
        stepSize     = {1,1},
        numSteps     = img_matrix:dim(),
        defaultValue = white,
        circular     = {false, false}
    }

    local dsImg = dataset.matrix(img_matrix, params_img)

    -- Create the indexed point
    -- the table is composed by elems (x, y, c)
    -- En la imagen es x, y i en la matriz y x
    local tIndexes = table.imap(table_points, function (elem)
        local index = elem[2]*img_matrix:dim(2)+elem[1] 
        assert(index <= img_matrix:size())
        return index
    end)
    local dsIndexes = dataset.matrix(matrix(tIndexes))

    local dsPoints = dataset.indexed(dsIndexes, {dsImg})

    local dsOut = nil
    local points
    if #table_points[1] >= 3 then
        points = table.imap(table_points, function (elem) return elem[3] end)
    end


    return dsPoints, points and interest_points.loadTagDataset(points, num_classes)
end
