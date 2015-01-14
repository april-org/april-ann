local MAX = 4

local function make_block_tostring(name, str)
  return function(b)
    local result = {}
    local data = {}
    for i=1,math.min(MAX, b:size()) do
      data[i] = str(b:get(i))
    end
    if b:size() > MAX then
      data[#data+1] = "..."
      data[#data+1] = b:get(b:size())
    end
    result[1] = table.concat(data, " ")
    result[2] = "# %s size %d [%s]"%{name, b:size(), b:get_reference_string()}
    return table.concat(result, "\n")
  end
end

mathcore.block.float.meta_instance.__tostring =
  make_block_tostring("Float block",
                      function(value)
                        return string.format("% -13.6g", value)
  end)

mathcore.block.float.meta_instance.__tostring =
  make_block_tostring("Double block",
                      function(value)
                        return string.format("% -15.6g", value)
  end)

mathcore.block.int32.meta_instance.__tostring =
  make_block_tostring("Int32 block",
                      function(value)
                        return string.format("% 11d", value)
  end)
