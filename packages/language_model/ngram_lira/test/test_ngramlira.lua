local path = arg[0]:dirname()
local filename = path .. "dihana3gram.lira.gz"
local check = utest.check
local T = utest.test
local IT = iterator

T("NoBackoffIteratorTest", function()
    local function get_expected_words(context)
      local context = context:gsub("(.)", "%%%1")
      local words = {}
      for line in io.lines(filename) do
        local word = line:match("%s+[^%s]+%s+" .. context .. "%s+([^%s]+)%s+[^%s]+%s+$")
        words[#words+1] = word
      end
      return words
    end

    local vocab = lexClass.load(io.open(path .. "vocab"))
    local model = language_models.load(filename, vocab, "<s>", "</s>")
    local lmi = cast.to( model:get_interface(), ngram.lira.interface )
    local k = lmi:get_initial_key()
    local words = IT(lmi:non_backoff_arcs_iterator(k):iterate()):table()
    local expected = get_expected_words("<s>")

    table.sort(words)
    table.sort(expected)
    
    check.TRUE(IT.zip(IT(words),IT(expected)):map(math.eq):reduce(math.land,true))
end)
