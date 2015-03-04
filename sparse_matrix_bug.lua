local m = matrix.fromFilename("/Users/pakozm/Dropbox/dev.april.mat.gz")
local s = matrix.sparse(m)
local ds = dataset.token.sparse_matrix(s)

s:toFilename("jarl.smat", "ascii")
local s2 = matrix.sparse.fromFilename("jarl.smat")

print(s == s2)

print(ds:numPatterns())
print(ds:patternSize())

--for ipat,pat in ds:patterns() do
--    print(ipat, pat)
--end

print(ds:getPatternBunch{1,2,4,5})
