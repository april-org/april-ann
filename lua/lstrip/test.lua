local tokens = [[
     and       break     do        else      elseif    end
     false     for       function  goto      if        in
     local     nil       not       or        repeat    return
     then      true      until     while
     +     -     *     /     %     ^     #
     &     ~     |     <<    >>    //
     ==    ~=    <=    >=    <     >     =
     (     )     {     }     [     ]     ::
     ;     :     ,     .     ..    ...
     0     0.1   -0.1  1.    .1
     name  'string'
]]

for v in tokens:gmatch("%S+") do
for w in tokens:gmatch("%S+") do
	print(v,w)
end
end
