-- FIXME: This test cannot be finished until we remember a corner case in which
-- lira model simplifies arpa specification

local path = arg[0]:dirname()
local filename = path .. "dihana3gram"
local check = utest.check
local T = utest.test
local IT = iterator

T("NoBackoffIteratorTest", function()
    local function get_expected_words(...)
      local context = IT(table.pack(...)):
        map(bind(string.gsub, nil, "([<>,.-+?])", "%%%1")):
        select(1):concat("%s+")
      local words = {}
      for line in io.lines(filename .. ".arpa") do
        local word = ( line:match("^%s*%S+%s+" .. context .. "%s+(%S+)%s+%S+%s*$") or
                         line:match("^%s*%S+%s+" .. context .. "%s+(%S+)%s*$") )
        if word and not tonumber(word) then words[#words+1] = word end
      end
      return words
    end

    local vocab = lexClass.load(io.open(path .. "vocab"))
    local model = language_models.load(filename .. ".lira.gz", vocab, "<s>", "</s>")
    local lmi = cast.to( model:get_interface(), ngram.lira.interface )
    local k = lmi:get_initial_key()
    local words = IT(lmi:non_backoff_arcs_iterator(k):iterate()):
      map(bind(vocab.getWordFromWordId, vocab)):table()
    local expected = get_expected_words("<s>")
    
    print( IT.zip(IT(words),IT(expected)):concat(" ", "\n") )
    
    table.sort(words)
    table.sort(expected)
    
    print(#words, #expected)
    
    check.eq(#words, #expected)
    check.TRUE(IT.zip(IT(words),IT(expected)):map(function(...) print(math.eq(...)) return ... end):map(math.eq):reduce(math.land,true))
end)
