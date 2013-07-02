interest_points = interest_points or {}

interest_points.pointClassifier = interest_points.pointClassifier or {}
interest_points.pointClassifier.__index = interest_points.pointClassifier

setmetatable(interest_points.pointClassifier, interest_points.pointClassifier)

--[[
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
]]

function interest_points.pointClassifier:__call(ancho, alto, miniancho, minialto, reverse)

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
    obj.tlc = dataset.linear_comb_conf(tlc)
    obj.white = (reverse and 0) or 1
    setmetatable(obj, self)
    return obj

end

---------------------------------------------------
-- crop_image(img, x, y, tlc) -> table
---------------------------------------------------
function interest_points.pointClassifier:crop_image(img, x, y)
    return self:applyLinearComb(img, x, y):getPattern(1)
end

function interest_points.pointClassifier:applyLinearComb(img, x, y)
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
function interest_points.pointClassifier:compute_point(img, point, mlp)
    local x, y = unpack(point)
    local data = self:crop_image(img,x,y)
    local salida = mlp:calculate(data)
    return salida 
end

-----------------------------------------------
-- Given a point return the most probable class
-----------------------------------------------
function interest_points.pointClassifier:classify_point(img, point, mlp)
    --return  argmax(self:compute_point(img, point, mlp))
    return compute_point(img, point, mlp):max()

end
-------------------------------------------------
-- Return a table with the triplets (x, y, class)
-------------------------------------------------
function interest_points.pointClassifier:compute_points(img, points_table, mlp)
  
    local res = {}
    for i, point in ipairs(points_table) do
      table.insert(res,self:compute_point(img, point, mlp))
    end

    return res
end

----------------------------------------------------------
-- Given a image extract all the points and classify them
-- -------------------------------------------------------
function interest_points.pointClassifier:classify_points(img, points, mlp)
      local scores = self:compute_points(img, points, mlp)
      local res = {}
      for i, score in ipairs(scores) do
          x = points[i][1]
          y = points[i][2]
          _, c = score:max()
          table.insert(res, {x, y, c})
      end

      return res
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
