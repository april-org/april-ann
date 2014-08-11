local check = utest.check
local T = utest.test
local result = { 1, 1, 1, 5, 5, 5 }

local mfset = util.mfset()

mfset:merge(1,2)
mfset:merge(2,3)
mfset:merge(4,6)
mfset:merge(5,6)

T("FindTest",
  function()
    -- mfset:print()
    for i=1,mfset:size() do
      check.eq( mfset:find(i), result[i] )
    end
end)

T("Serialization",
  function()
    local t=util.mfset.fromString(mfset:toString())
    -- t:print()
    for i=1,t:size() do
      check.eq( t:find(i), mfset:find(i) )
    end
end)

