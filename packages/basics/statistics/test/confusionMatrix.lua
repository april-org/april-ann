-- Check 4 clases with table

local N = 4
local myConf = stats.confusion_matrix(N)
print (myConf)
local myPred = { 1, 2, 3, 4, 1, 2 , 3 , 4}
local myGT   = { 2, 1, 2, 1, 1, 2,  3,  1}

myConf:addData(myPred, myGT)
print("Raw Confusion Matrix")
myConf:printConfusion()


myConf:reset()
print("Empty Matrix")
myConf:printConfusion()

local t = {}
for i, v in ipairs(myPred) do
  table.insert(t, {v,myGT[i]})
end
myConf:addData(stats.confusion_matrix.oneTableIterator(t))

print("Raw Confusion Matrix")
myConf:printConfusion()

print("Results")

print("Error ", myConf:getError())
print("Accuracy ", myConf:getAccuracy())

for i = 1, N do
  print(i, "PR:", myConf:getPrecision(i), "RC:", myConf:getRecall(i),"FM:", myConf:getFMeasure(i))
end

print("Testing Datasets Methods")

local dsPred = dataset.matrix(matrix(#myPred, myPred))
local dsGT = dataset.matrix(matrix(#myGT, myGT))

myConf:reset()
myConf:addData(stats.confusion_matrix.twoDatasetsIterator(dsPred,dsGT))
print("Raw Confusion Matrix")
myConf:printConfusion()

myConf:reset()
local dsOne = dataset.join({dsPred, dsGT})
myConf:addData(stats.confusion_matrix.oneDatasetIterator(dsOne))

print("Raw Confusion Matrix")
myConf:printConfusion()

