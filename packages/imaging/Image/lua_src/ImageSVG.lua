local imageSVG,imageSVG_methods = class("imageSVG")
_G.imageSVG = imageSVG -- global environment

local colors = { "red", "blue", "green", "orange", "purple","yellow"}

function toImage64(filename)

 local f = io.open(filename)
 base64 = require("ee5_base64") 
 local s64 = base64.encode(f:read("*all"))
 f:close()
 return s64
end

function imageSVG:constructor(params)
  
    self.width = params.width or -1
    self.height = params.height or -1

    -- Header, body and footer are tables of strings
    self.header = {}
    self.body   = {}
    self.footer = {}
    
end

function imageSVG_methods:setHeader()

    self.header = {}

    --  local swidth  = tostring(self.width) or "100%"
    --  local sheight = tostring(self.height) or "100%"

    local swidth  = (self.width ~= -1 and tostring(self.width)) or "100%"
    local sheight = (self.height ~= -1 and tostring(self.height)) or "100%"
    table.insert(self.header,string.format("<svg width=\"%s\" height=\"%s\"\n xmlns=\"http://www.w3.org/2000/svg\"  xmlns:xlink=\"http://www.w3.org/1999/xlink\" >\n", swidth, sheight))
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

function imageSVG_methods:addPolygon(points, params)

    local id = params.id or ""
    local stroke = params.stroke or "black"
    local stroke_width = params.stroke_width or "2"

    local buffer = {}

    table.insert(buffer, string.format("<polygon id=\"%s\" points =\"", id))
    for i, v in ipairs(points) do
        table.insert(buffer, string.format("%0.4f, %0.4f ", v[1], v[2]))
    end

    table.insert(buffer, string.format("\" style=\"fill:none;stroke:%s;stroke-width:%s;\"/>", stroke, stroke_width))

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


-- Each element of the table is a path
function imageSVG_methods:addInterestPointPaths(paths, ...)

    params = table.pack(...)
    local num_classes = params.num_class or 5
    for i, path in ipairs(paths) do
        -- Process the component
        --
        local color = "black"
        color = colors[i%#colors+1]

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
    local opacity = params.opacity or 1
    local cx = point[1]
    local cy = point[2]
    table.insert(self.body,string.format("<circle cx=\"%d\" cy=\"%d\" r=\"%f\" fill=\"%s\" fill-opacity=\"%f\" />",
    cx, cy, radius, color, opacity))
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

function imageSVG_methods:addRect(rect, params)

    local color = params.color or "black"

    if params.cls then
        color = colors[params.cls]
    end

    local x = rect[1]
    local y = rect[2]
    local w = rect[3] - rect[1]
    local h = rect[4] - rect[2]
    table.insert(self.body,string.format("<rect x=\"%d\" y=\"%d\" width=\"%f\" height=\"%f\" stroke=\"%s\" fill-opacity=\"0\"/> ", x, y, w, h, color))
end

function imageSVG_methods:addPoint(point, params)
    -- TODO: add confidence

    if params.conf then 
        self:addCircle({point[1],point[2]}, {color = colors[params.cls] , radius = (params.conf*2)^2, opacity = 0.5})
    end

    if params.circle then
        self:addCircle({point[1],point[2]}, {color = colors[params.cls] , radius = params.side})
    else
        self:addSquare(point, params)
    end
end

function imageSVG_methods:addTriangle(point, params)


    local side = params.side or 1

    if params.reverse then
        side = -side
    end
    local mid = point[2] - side
    local color = params.color or "black"
    local leftx, lefty = point[1]-side, point[2]+side
    local rightx, righty = point[1]+side, point[2]+side


    local spoints = string.format("%f,%f %f,%f %f,%f", point[1], mid, leftx, lefty, rightx, righty)

    table.insert(self.body, string.format('<polygon points="%s" style="fill:%s"/>', spoints, color)) 
end

function imageSVG_methods:addLine(ini_point, end_point, params)

    --local color = params.color or "black"
    table.insert(self.body, string.format('<line x1="%d" y1="%d" x2="%d" y2="%d"/>', 
    ini_point[1], ini_point[2], end_point[1], end_point[2]))

end
-- Extra image function
function imageSVG_methods:addImage(filename, width, height, offsetX, offsetY, absolute)


    absolute  = absolute or false

    if absolute then
        filename = os.getenv("PWD").."/"..filename
    end
    table.insert(self.body, string.format("<image x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" xlink:href=\"%s\">\n</image>", offsetX, offsetY, width, height, filename))

end

function imageSVG_methods:addEmbedImage(filename, width, height, offsetX, offsetY)

    local image64 = toImage64(filename)

    table.insert(self.body, string.format("<image x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\"\
    xlink:href=\"data:image/png;base64,%s \"/>", offsetX, offsetY, width, height, image64))

end
function imageSVG.fromImageFile(filename, width, height, absolute)

    mySVG = imageSVG({width = width, height = height})
    mySVG:setHeader()
    mySVG:setFooter()

    absolute = absolute or false

    mySVG:addImage(filename, width, height, 0, 0, absolute)
    return mySVG  
end
