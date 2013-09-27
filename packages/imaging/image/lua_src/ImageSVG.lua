local imageSVG_methods,
imageSVG_class_metatable = class("imageSVG")

local colors = { "red", "blue", "green", "orange", "purple"}

function imageSVG_class_metatable:__call(params)

    local obj = {}

    obj.width = params.width or -1
    obj.height = params.height or -1

    -- Header, body and footer are tables of strings
    obj.header = {}
    obj.body   = {}
    obj.footer = {}
    
    obj = class_instance(obj, self)

    return obj

end

function imageSVG_methods:setHeader()

    self.header = {}
    table.insert(self.header, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    table.insert(self.header, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 20001102//EN\" \"http://www.w3.org/TR/2000/CR-SVG-20001102/DTD/svg-20001102.dtd\">")

    --  local swidth  = tostring(self.width) or "100%"
    --  local sheight = tostring(self.height) or "100%"

    local swidth  = (self.width ~= -1 and tostring(self.width)) or "100%"
    local sheight = (self.height ~= -1 and tostring(self.height)) or "100%"
    table.insert(self.header,string.format("<svg width=\"%s\" height=\"%s\"\nxmlns:xlink=\"http://www.w3.org/1999/xlink\" >\n", swidth, sheight))
end

function imageSVG_methods:setFooter()
    self.footer = {}
    table.insert(self.footer, "</svg>")
end

-- Recieves a table with points x,y
function imageSVG_methods:addPathFromTable(path, params)
    --params treatment

    local id = params.id or ""
    local stroke = params.stroke or "black"
    local stroke_width = params.stroke_width or "2"

    local buffer = {}

    table.insert(buffer, string.format("<path id = \"%s\" d = \"", id))

    for i, v in ipairs(path) do
        if (i == 1) then
            table.insert(buffer, string.format("M %d %d ", v[1], v[2]))
        else
            table.insert(buffer, string.format("L %d %d ", v[1], v[2]))
        end

    end
    table.insert(buffer, string.format("\" fill=\"none\" stroke = \"%s\" stroke-width = \"%s\"/>", stroke, stroke_width))

    table.insert(self.body, table.concat(buffer))
end

function imageSVG_methods:getHeader()
    return table.concat(self.header, "\n") 
end
function imageSVG_methods:getBody()
    return table.concat(self.body, "\n") 
end
function imageSVG_methods:getFooter()
    return table.concat(self.footer, "\n") 
end
function imageSVG_methods:getString()
    return table.concat({self:getHeader(), self:getBody(), self:getFooter()})
end
function imageSVG_methods:write(filename)
    file = io.open(filename, "w")
    file:write(self:getHeader())
    file:write(self:getBody())
    file:write(self:getFooter())
end

-- Each element of the table is a path
function imageSVG_methods:addPaths(paths)
    for i, path in ipairs(paths) do
        local color = "black"
            color = colors[i%#colors+1]
        --        print(i, color, #path)
        self:addPathFromTable(path,{stroke = color, id = tostring(i)})
    end
end



-- Given a table with table of points, draw
function imageSVG_methods:addPointsFromTables(tables, size)

    for i, points in ipairs(tables) do
        local color = "black"
        if (i <= #colors) then
            color = colors[i]
        end

        for j, point in ipairs(points) do
            self:addPoint(point, {color = color, side = size})
        end
    end
end
function imageSVG_methods:resize(height, width)
    self.height = height
    self.width = width
end

-- Point is a table with two coordinates
function imageSVG_methods:addCircle(point, params)

    local radius = params.radius or 1
    local color = params.color or "green"

    local cx = point[1]
    local cy = point[2]
    table.insert(self.body,string.format("<circle cx=\"%d\" cy=\"%d\" r=\"%f\" fill=\"%s\"/>", cx, cy, radius, color))
end

-- Point is a table with two coordinates
function imageSVG_methods:addSquare(point, params)

    local side = params.side or 1
    local color = params.color or "green"
    
    if params.cls then
      color = colors[params.cls]
    end
    local x = point[1]
    local y = point[2]

    table.insert(self.body,string.format("<rect x=\"%d\" y=\"%d\" width=\"%f\" height=\"%f\" fill=\"%s\"/>", x, y, side,side, color))
end

function imageSVG_methods:addPoint(point, params)
    self:addSquare(point, params)
end

-- Extra image function
function imageSVG_methods:addImage(filename, width, height, offsetX, offsetY)

    table.insert(self.body, string.format("<image x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" xlink:href=\"%s\">\n</image>", offsetX, offsetY, width, height, filename))

end

function imageSVG.fromImageFile(filename, width, height)

    mySVG = imageSVG({width = width, height = height})
    mySVG:setHeader()
    mySVG:setFooter()
    mySVG:addImage(filename, width, height, 0, 0)
    return mySVG  
end
