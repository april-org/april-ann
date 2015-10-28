-- Check 4 clases with table
local N = 4
local myConf = stats.confusion_matrix(N)
local myPred = { 1, 2, 3, 4, 1, 2 , 3 , 4}
local myGT   = { 2, 1, 2, 1, 1, 2,  3,  1}

myConf:addData(myPred, myGT)
print("Raw Confusion Matrix")
myConf:printConfusion()
printf("#\n----------------------#\n")
myConf:reset()
print("Empty Matrix")
myConf:printConfusion()

printf("#\n----------------------#\n")
local t = {}
for i, v in ipairs(myPred) do
  table.insert(t, {v,myGT[i]})
end
myConf:addData(stats.confusion_matrix.oneTableIterator(t))

print("Raw Confusion Matrix")
myConf:printConfusion()

printf("#\n----------------------#\n")
print("Results")

print("Error ", myConf:getError())
print("Accuracy ", myConf:getAccuracy())
print("AverageError", myConf:getAvgError())
printf("#\n----------------------#\n")

print("Weighted Error 0.5 0.1 0.1 0.3", myConf:getWeightedError({0.5, 0.1, 0.1, 0.3}))
for i = 1, N do
  print(i, "PR:", myConf:getPrecision(i), "RC:", myConf:getRecall(i),"FM:", myConf:getFMeasure(i))
end

printf("#\n----------------------#\n")
print ("Joining classes: 1, 2")
print("New recall (1 and 2)", myConf:getMultiRecall{1,2})
print("New precision (1 an 2) ", myConf:getMultiPrecision{1,2})

printf("#\n----------------------#\n")

print("Testing Datasets Methods")

local dsPred = dataset.matrix(matrix(#myPred, myPred))
local dsGT = dataset.matrix(matrix(#myGT, myGT))

myConf:reset()
myConf:addData(stats.confusion_matrix.twoDatasetsIterator(dsPred,dsGT))
print("Raw Confusion Matrix")
myConf:printConfusion()

myConf:reset()
printf("#\n----------------------#\n")
local dsOne = dataset.join({dsPred, dsGT})
myConf:addData(stats.confusion_matrix.oneDatasetIterator(dsOne))

print("Raw Confusion Matrix")
myConf:printConfusion()

printf("#\n----------------------#\n")
print("Testing Map")

local tags = {"a","b","c","d"}
local myConfMap = stats.confusion_matrix(4, table.invert(tags))

local myPred = { "a", "b", "c", "d", "a", "b" , "c" , "d"}
local myGT   = { "b", "a", "b", "a", "a", "b",  "c",  "a"}

myConfMap:addData(myPred, myGT)

print("Raw Confusion Matrix")
myConfMap:printConfusion(myConfMap)
local cloneConf = myConfMap:clone()

printf("#\n----------------------#\n")
print("Confusion Matrix String")

print(myConfMap:tostring())
printf("#\n----------------------#\n")
print("Deleting class c")
myConfMap:clearGTClass(3)
myConfMap:printConfusion(tags)
printf("#\n----------------------#\n")
print("Cloned object")
cloneConf:printConfusion(tags)
printf("#\n----------------------#\n")
print("Deleting class c (Pred)")
cloneConf:clearPredClass(3)
cloneConf:printConfusion(tags)

print("Matrix")

m1 = matrix{0, 1, 0, 1}
m2 = matrix{0, 1, 1, 0}

local m_confusion = stats.confusion_matrix(2)
m_confusion:addData(m1,m2)
m_confusion:printConfusion()
