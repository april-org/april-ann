
mySVG = imageSVG({})
mySVG:setHeader({})
mySVG:setFooter({})

print(mySVG:getString())

t = { {0,0}, {100,0}, {100,100}, {0, 100} }
mySVG:addPathFromTable(t, { stroke = "red"})

t = { {0,0}, {50,50}, {100,0}, {0, 100} }
mySVG:addPathFromTable(t, { stroke = "green"})
print("===============")
print(mySVG:getString())

mySVG:write("img1.svg")

print(mySVG:getString())

--New SVG
mySVG = imageSVG({width=300, height = 300})
mySVG:setHeader({})
mySVG:setFooter({})

t1 = { {0,0}, {100,0}, {100,100}, {0, 100} }
t2 = { {0,0}, {50,50}, {100,0}, {0, 100} }
t = {t1,t2}
mySVG:addImage("molo.png",100,100, 50, 50)
mySVG:addPaths(t)
mySVG:write("img2.svg")

