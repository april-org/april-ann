
wop = util.stopwatch()
wop:reset()

wop:go()
local a = 0
for i=0,10000 do
      a = a+1
end
wop:stop()

print("ha tardado:",wop:read())

if wop:is_on() then
   print("el cronometro esta en marcha")
else
   print("el cronometro esta parado")
end


wop2 = wop:clone()

wop2:go()
local a = 0
for i=0,500000000 do
      a = a+1
end
wop2:stop()
print("ha tardado:",wop2:read())


