local check = utest.check
local T = utest.test

T("SparseMatrixDataSet",
  function()
    local ds = dataset.token.sparse_matrix(matrix.sparse.diag{6,5,4,3,2,1})
    check.eq( ds:getPattern(1),
              matrix.sparse(matrix(1, 6, { 6, 0, 0, 0, 0, 0 })) )
    check.eq( ds:getPattern(2),
              matrix.sparse(matrix(1, 6, { 0, 5, 0, 0, 0, 0 })) )
    check.eq( ds:getPattern(3),
              matrix.sparse(matrix(1, 6, { 0, 0, 4, 0, 0, 0 })) )
    check.eq( ds:getPattern(4),
              matrix.sparse(matrix(1, 6, { 0, 0, 0, 3, 0, 0 })) )
    check.eq( ds:getPattern(5),
              matrix.sparse(matrix(1, 6, { 0, 0, 0, 0, 2, 0 })) )
    check.eq( ds:getPattern(6),
              matrix.sparse(matrix(1, 6, { 0, 0, 0, 0, 0, 1 })) )
    check.eq( ds:getPatternBunch{6, 1, 3},
              matrix.sparse(matrix(3, 6, { 0, 0, 0, 0, 0, 1,
                                           6, 0, 0, 0, 0, 0,
                                           0, 0, 4, 0, 0, 0, })) )
    local uds = dataset.token.union{ ds, ds }
    check.eq( uds:getPattern(4),
              matrix.sparse(matrix(1, 6, { 0, 0, 0, 3, 0, 0 })) )
    check.eq( uds:getPattern(11),
              matrix.sparse(matrix(1, 6, { 0, 0, 0, 0, 2, 0 })) )
    check.eq( uds:getPatternBunch{6, 7, 3},
              matrix.sparse(matrix(3, 6, { 0, 0, 0, 0, 0, 1,
                                           6, 0, 0, 0, 0, 0,
                                           0, 0, 4, 0, 0, 0, })) )    
end)
