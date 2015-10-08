--[[**************************************************************************]]
-- base64.lua
-- Copyright 2014 Ernest R. Ewert
--
--  This Lua module contains the implementation of a Lua base64 encode
--  and decode library.
--
--  The library exposes these methods.
--
--      Method      Args
--      ----------- ----------------------------------------------
--      encode      String in / out
--      decode      String in / out
--
--      encode      String, function(value) predicate
--      decode      String, function(value) predicate
--
--      encode      file, function(value) predicate
--      deocde      file, function(value) predicate
--
--      encode      file, file
--      deocde      file, file
--
--      alpha       alphabet, term char
--

if package.loaded["ee5_base64"] then
  local ee5_base64 = require "ee5_base64"
  _G.ee5_base64 = ee5_base64
  return ee5_base64
end

--------------------------------------------------------------------------------
-- known_base64_alphabets
--
--
local known_base64_alphabets=
  {
    base64= -- RFC 2045 (Ignores max line length restrictions)
      {
        _alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
        _strip="[^%a%d%+%/%=]",
        _term="="
      },

    base64noterm= -- RFC 2045 (Ignores max line length restrictions)
      {
        _alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
        _strip="[^%a%d%+%/]",
        _term=""
      },

    base64url= -- RFC 4648 'base64url'
      {
        _alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
        _strip="[^%a%d%+%-%_=]",
        _term=""
      },
  }
local c_alpha=known_base64_alphabets.base64
local pattern_strip

--[[**************************************************************************]]
--[[****************************** Encoding **********************************]]
--[[**************************************************************************]]


--------------------------------------------------------------------------------
-- base64 encoding table
--
-- Each (zero based, six bit) index is matched against the ASCII
-- value that represents the six bit pattern.
--
--          [6 bit encoding]=ASCII value
--
-- This table varies from normal Lua one based indexing to avoid
-- extra math during the fix-ups. This is a performance improvement
-- for very long encoding runs.
--
local b64e=
  {
    [ 0]= 65, [ 1]= 66, [ 2]= 67, [ 3]= 68, [ 4]= 69, [ 5]= 70,
    [ 6]= 71, [ 7]= 72, [ 8]= 73, [ 9]= 74, [10]= 75, [11]= 76,
    [12]= 77, [13]= 78, [14]= 79, [15]= 80, [16]= 81, [17]= 82,
    [18]= 83, [19]= 84, [20]= 85, [21]= 86, [22]= 87, [23]= 88,
    [24]= 89, [25]= 90, [26]= 97, [27]= 98, [28]= 99, [29]=100,
    [30]=101, [31]=102, [32]=103, [33]=104, [34]=105, [35]=106,
    [36]=107, [37]=108, [38]=109, [39]=110, [40]=111, [41]=112,
    [42]=113, [43]=114, [44]=115, [45]=116, [46]=117, [47]=118,
    [48]=119, [49]=120, [50]=121, [51]=122, [52]= 48, [53]= 49,
    [54]= 50, [55]= 51, [56]= 52, [57]= 53, [58]= 54, [59]= 55,
    [60]= 56, [61]= 57, [62]= 43, [63]= 47
  }
-- Precomputed tables (compromise using more memory for speed)
local b64e_a  -- ready to use
local b64e_a2 -- byte addend
local b64e_b1 -- byte addend
local b64e_b2 -- byte addend
local b64e_c1 -- byte addend
local b64e_c  -- ready to use


-- Tail padding values
local tail_padd64=
  {
    "==",   -- two bytes modulo
    "="     -- one byte modulo
  }


--------------------------------------------------------------------------------
-- m64
--
--  Helper function to convert three eight bit values into four encoded
--  6 (significant) bit values.
--
--                 7             0 7             0 7             0
--             m64(a a a a a a a a,b b b b b b b b,c c c c c c c c)
--                 |           |           |           |
--  return    [    a a a a a a]|           |           |
--                        [    a a b b b b]|           |
--                                    [    b b b b c c]|
--                                                [    c c c c c c]
--
local ext = bit32.extract -- slight speed, vast visual (IMO)

local function m64( a, b, c )
  -- Return pre-calculated values for encoded value 1 and 4
  -- Get the pre-calculated extractions for value 2 and 3, look them
  -- up and return the proper value.
  --
  return  b64e_a[a],
  b64e[ b64e_a2[a]+b64e_b1[b] ],
  b64e[ b64e_b2[b]+b64e_c1[c] ],
  b64e_c[c]
end

--------------------------------------------------------------------------------
-- encode_tail64
--
--  Send a tail pad value to the output predicate provided.
--
local function encode_tail64( out, x, y )
  -- If we have a number of input bytes that isn't exactly divisible
  -- by 3 then we need to pad the tail
  if x ~= nil then
    local a,b,r = x,0,1

    if y ~= nil then
      r = 2
      b = y
    end

    -- Encode three bytes of info, with the tail byte as zeros and
    -- ignore any fourth encoded ASCII value. (We should NOT have a
    -- forth byte at this point.)
    local b1, b2, b3 = m64( a, b, 0 )

    -- always add the first 2 six bit values to the res table
    -- 1 remainder input byte needs 8 output bits
    local tail_value = string.char( b1, b2 )

    -- two remainder input bytes will need 18 output bits (2 as pad)
    if r == 2 then
      tail_value=tail_value..string.char( b3 )
    end

    -- send the last 4 byte sequence with appropriate tail padding
    out( tail_value .. tail_padd64[r] )
  end
end


--------------------------------------------------------------------------------
-- encode64_io_iterator
--
--  Create an io input iterator to read an input file and split values for
--  proper encoding.
--
local function encode64_io_iterator(file)

  assert( io.type(file)  == "file" or class.is_a(file,aprilio.stream),
          "argument must be readable file handle" )
  assert( file.read ~= nil, "argument must be readable file handle" )

  local ii = { } -- Table for the input iterator

  setmetatable(ii,{ __tostring=function() return "base64.io_iterator" end})

  -- Begin returns an input read iterator
  --
  function ii.begin()
    local sb  = string.byte

    -- The iterator returns three bytes from the file for encoding or nil
    -- when the end of the file has been reached.
    --
    return function()
      s = file:read(3)
      if s ~= nil and #s == 3 then
        return sb(s,1,3)
      end
      return nil
    end
  end

  -- The tail method on the iterator allows the routines to run faster
  -- because each sequence of bytes doesn't have to test for EOF.
  --
  function ii.tail()
    -- If one or two "overflow" bytes exist, return those.
    --
    if s ~= nil then return s:byte(1,2) end
  end

  return ii
end


--------------------------------------------------------------------------------
-- encode64_with_ii
--
--      Convert the value provided by an encode iterator that provides a begin
--      method, a tail method, and an iterator that returns three bytes for
--      each call until at the end. The tail method should return either 1 or 2
--      tail bytes (for source values that are not evenly divisible by three).
--
local function encode64_with_ii( ii, out )
  local sc=string.char

  for a, b, c in ii.begin() do
    out( sc( m64( a, b, c ) ) )
  end

  encode_tail64( out, ii.tail() )

end


--------------------------------------------------------------------------------
-- encode64_with_predicate
--
--      Implements the basic raw data --> base64 conversion. Each three byte
--      sequence in the input string is converted to the encoded string and
--      given to the predicate provided in 4 output byte chunks. This method
--      is slightly faster for traversing existing strings in memory.
--
local function encode64_with_predicate( raw, out )
  local rem=#raw%3     -- remainder
  local len=#raw-rem   -- 3 byte input adjusted
  local sb=string.byte -- Mostly notational (slight performance)
  local sc=string.char -- Mostly notational (slight performance)

  -- Main encode loop converts three input bytes to 4 base64 encoded
  -- ACSII values and calls the predicate with the value.
  for i=1,len,3 do
    -- This really isn't intended as obfuscation. It is more about
    -- loop optimization and removing temporaries.
    --
    out( sc( m64( sb( raw ,i , i+3 ) ) ) )
    --   |   |    |
    --   |   |    byte i to i + 3
    --   |   |
    --   |   returns 4 encoded values
    --   |
    --   creates a string with the 4 returned values
  end

  -- If we have a number of input bytes that isn't exactly divisible
  -- by 3 then we need to pad the tail
  if rem > 0 then
    local x, y = sb( raw, len+1 )

    if rem > 1 then
      y = sb( raw, len+2 )
    end

    encode_tail64( out, x, y )
  end
end


--------------------------------------------------------------------------------
-- encode64_tostring
--
--      Convenience method that accepts a string value and returns the
--      encoded version of that string.
--
local function encode64_tostring(raw)

  local sb={} -- table to build string

  local function collection_predicate(v)
    sb[#sb+1]=v
  end

  -- Test with an 818K string in memory. Result is 1.1M of data.
  --
  --      lua_base64      base64 (gnu 8.21)
  --      202ms           54ms
  --      203ms           48ms
  --      204ms           50ms
  --      203ms           42ms
  --      205ms           46ms
  --
  encode64_with_predicate( raw, collection_predicate )

  return table.concat(sb)
end


--[[**************************************************************************]]
--[[****************************** Decoding **********************************]]
--[[**************************************************************************]]


--------------------------------------------------------------------------------
-- base64 decoding table
--
-- Each ASCII encoded value index is matched against the zero based, six bit
-- bit pattern.
--
--          [ASCII value]=6 bit encoding value
--
local b64d=
  {
    [ 65]= 0, [ 66]= 1, [ 67]= 2, [ 68]= 3, [ 69]= 4, [ 70]= 5,
    [ 71]= 6, [ 72]= 7, [ 73]= 8, [ 74]= 9, [ 75]=10, [ 76]=11,
    [ 77]=12, [ 78]=13, [ 79]=14, [ 80]=15, [ 81]=16, [ 82]=17,
    [ 83]=18, [ 84]=19, [ 85]=20, [ 86]=21, [ 87]=22, [ 88]=23,
    [ 89]=24, [ 90]=25, [ 97]=26, [ 98]=27, [ 99]=28, [100]=29,
    [101]=30, [102]=31, [103]=32, [104]=33, [105]=34, [106]=35,
    [107]=36, [108]=37, [109]=38, [110]=39, [111]=40, [112]=41,
    [113]=42, [114]=43, [115]=44, [116]=45, [117]=46, [118]=47,
    [119]=48, [120]=49, [121]=50, [122]=51, [ 48]=52, [ 49]=53,
    [ 50]=54, [ 51]=55, [ 52]=56, [ 53]=57, [ 54]=58, [ 55]=59,
    [ 56]=60, [ 57]=61, [ 43]=62, [ 47]=63
  }
-- Precomputed tables (compromise using more memory for speed)
local b64d_a1 -- byte addend
local b64d_a2 -- byte addend
local b64d_b1 -- byte addend
local b64d_b2 -- byte addend
local b64d_c1 -- byte addend


--------------------------------------------------------------------------------
-- u64
--
--  Helper function to convert four six bit values into three full eight
--  bit values. Input values are the integer expression of the six bit value
--  encoded in the original base64 encoded string.
--
--     u64( _ _1 1 1 1 1 1,
--             |       _ _ 2 2 2 2 2 2,
--             |           |       _ _ 3 3 3 3 3 3,
--             |           |           |       _ _ 4 4 4 4 4 4)
--             |           |           |           |
--  return ', [1 1 1 1 1 1 2 2]        |           |
--         ',                 [2 2 2 2 3 3 3 3]    |
--         '                                  [3 3 4 4 4 4 4 4]
--
local function u64( b1, b2, b3, b4 )
  -- We can get away with addition instead of anding the values together
  -- because there are no  overlapping bit patterns.
  --
  return
    b64d_a1[b1] + b64d_a2[b2],
  b64d_b1[b2] + b64d_b2[b3],
  b64d_c1[b3] + b64d[b4]
end


--------------------------------------------------------------------------------
-- decode_tail64
--
--  Send the end of stream bytes that didn't get decoded via the main loop.
--
local function decode_tail64( out, e1, e2 ,e3, e4 )

  if tail_padd64[2] == "" or e4 == tail_padd64[2]:byte() then
    local n3 = b64e[0]

    if e3 ~= nil and e3 ~= tail_padd64[2]:byte() then
      n3 = e3
    end

    -- Unpack the six bit values into the 8 bit values
    local b1, b2 = u64( e1, e2, n3, b64e[0] )

    -- And add them to the res table
    if e3 ~= nil and e3 ~= tail_padd64[2]:byte() then
      out( string.char( b1, b2 ) )
    else
      out( string.char( b1 ) )
    end
  end
end


--------------------------------------------------------------------------------
-- decode64_io_iterator
--
--  Create an io input iterator to read an input file and split values for
--  proper decoding.
--
local function decode64_io_iterator( file )

  local ii = { }

  -- An enumeration coroutine that handles the reading of an input file
  -- to break data into proper pieces for building the original string.
  --
  local function enummerate( file )
    local sc=string.char
    local sb=string.byte
    local ll="" -- last line storage
    local len

    -- Read a "reasonable amount" of data into the line buffer. Line by
    -- line is not used so that a file with no line breaks doesn't
    -- cause an inordinate amount of memory usage.
    --
    for cl in file:lines(2048) do
      -- Reset the current line to contain valid chars and any previous
      -- "leftover" bytes from the previous read
      --
      cl = ll .. cl:gsub(pattern_strip,"")
      --   |     |
      --   |     Remove "Invalid" chars (white space etc)
      --   |
      --   Left over from last line
      --
      len = (#cl-4)-(#cl%4)

      -- see the comments in decode64_with_predicate for a rundown of
      -- the results of this loop (sans the coroutine)
      for i=1,len,4 do
        coroutine.yield
        (
          sc( u64( sb( cl, i, i+4 ) ) )
        )
      end

      ll = cl:sub( len +1, #cl )
    end

    local l = #ll

    if l >= 4 and ll:sub(-1) ~= tail_padd64[2] then
      coroutine.yield
      (
        sc( u64( sb( ll, 1, 4 ) ) )
      )
      l=l-4
    end

    if l > 0 then

      local e1,e2,e3,e4 = ll:byte( 0 - l, -1 )

      if e1 ~= nil then
        decode_tail64( function(s) coroutine.yield( s ) end, e1, e2, e3, e4 )
      end
    end

  end

  -- Returns an input iterator that is implemented as a coroutine. Each
  -- yield of the co-routine sends reconstructed bytes to the lopp handling
  -- the iteration.
  --
  function ii.begin()
    local co = coroutine.create( function() enummerate(file) end )

    return function()
      local code,res = coroutine.resume(co)
      assert(code == true)
      return res
    end
  end

  return ii
end


--------------------------------------------------------------------------------
-- decode64_with_ii
--
--      Convert the value provided by a decode iterator that provides a begin
--      method, a tail method, and an iterator that returns four (usable!) bytes
--      for each call until at the end.
--
local function decode64_with_ii( ii, out )

  -- Uses the iterator to pull values. Each reconstructed string
  -- is sent to the output predicate.
  --
  for l in ii.begin() do out( l ) end

end


--------------------------------------------------------------------------------
-- decode64_with_predicate
--
-- Decode an entire base64 encoded string in memory using the predicate for
-- output.
--
local function decode64_with_predicate( raw, out )
  -- Sanitize the input to strip characters that are not in the alphabet.
  --
  -- Note: This is a deviation from strict implementations where "bad data"
  --       in the input stream is unsupported.
  --
  local san = raw:gsub(pattern_strip,"")
  local len = #san-#san%4         --
  local rem = #san-len
  local sc  = string.char
  local sb  = string.byte

  if san:sub(-1,-1) == tail_padd64[2] then
    rem = rem + 4
    len = len - 4
  end

  for i=1,len,4 do
    out( sc( u64( sb( san, i, i+4 ) ) ) )
  end

  if rem > 0 then
    decode_tail64( out, sb( san, 0-rem, -1 ) )
  end
end


--------------------------------------------------------------------------------
-- decode64_tostring
--
--  Takes a string that is encoded in base64 and returns the decoded value in
--  a new string.
--
local function decode64_tostring( raw )

  local sb={} -- table to build string

  local function collection_predicate(v)
    sb[#sb+1]=v
  end

  decode64_with_predicate( raw, collection_predicate )

  return table.concat(sb)
end


--------------------------------------------------------------------------------
-- set_and_get_alphabet
--
--  Sets and returns the encode / decode alphabet.
--
--
local function set_and_get_alphabet(alpha,term)

  if alpha ~= nil then
    local magic=
      {
        --        ["%"]="%%",
        [" "]="% ",
        ["^"]="%^",
        ["$"]="%$",
        ["("]="%(",
        [")"]="%)",
        ["."]="%.",
        ["["]="%[",
        ["]"]="%]",
        ["*"]="%*",
        ["+"]="%+",
        ["-"]="%-",
        ["?"]="%?",
      }

    c_alpha=known_base64_alphabets[alpha]
    if c_alpha == nil then
      c_alpha={ _alpha=alpha, _term=term }
    end

    assert( #c_alpha._alpha == 64,    "The alphabet ~must~ be 64 unique values."  )
    assert( #c_alpha._term  <=  1,    "Specify zero or one termination character.")

    b64d={}
    b64e={}
    local s=""
    for i = 1,64 do
      local byte = c_alpha._alpha:byte(i)
      local str  = string.char(byte)
      b64e[i-1]=byte
      assert( b64d[byte] == nil, "Duplicate value '"..str.."'" )
      b64d[byte]=i-1
      s=s..str
    end

    -- preload encode lookup tables
    b64e_a  = {}
    b64e_a2 = {}
    b64e_b1 = {}
    b64e_b2 = {}
    b64e_c1 = {}
    b64e_c  = {}

    for f = 0,255 do
      b64e_a  [f]=b64e[ext(f,2,6)]
      b64e_a2 [f]=ext(f,0,2)*16
      b64e_b1 [f]=ext(f,4,4)
      b64e_b2 [f]=ext(f,0,4)*4
      b64e_c1 [f]=ext(f,6,2)
      b64e_c  [f]=b64e[ext(f,0,6)]
    end

    -- preload decode lookup tables
    b64d_a1 = {}
    b64d_a2 = {}
    b64d_b1 = {}
    b64d_b2 = {}
    b64d_c1 = {}

    for k,v in pairs(b64d) do
      -- Each comment shows the rough C expression that would be used to
      -- generate the returned triple.
      --
      b64d_a1 [k] = v*4                   -- ([b1]       ) << 2
      b64d_a2 [k] = math.floor( v / 16 )  -- ([b2] & 0x30) >> 4
      b64d_b1 [k] = ext( v, 0, 4 ) * 16   -- ([b2] & 0x0F) << 4
      b64d_b2 [k] = math.floor( v / 4 )   -- ([b3] & 0x3c) >> 2
      b64d_c1 [k] = ext( v, 0, 2 ) * 64   -- ([b3] & 0x03) << 6
    end

    if c_alpha._term ~= "" then
      tail_padd64[1]=string.char(c_alpha._term:byte(),c_alpha._term:byte())
      tail_padd64[2]=string.char(c_alpha._term:byte())
    else
      tail_padd64[1]=""
      tail_padd64[2]=""
    end

    local esc_term

    if magic[c_alpha._term] ~= nil then
      esc_term=c_alpha._term:gsub(magic[c_alpha._term],function (s) return magic[s] end)
    elseif c_alpha._term == "%" then
      esc_term = "%%"
    else
      esc_term=c_alpha._term
    end

    if not c_alpha._strip then
      local p=s:gsub("%%",function (s) return "__unique__" end)
      for k,v in pairs(magic)
      do
        p=p:gsub(v,function (s) return magic[s] end )
      end
      local mr=p:gsub("__unique__",function() return "%%" end)

      c_alpha._strip = string.format("[^%s%s]",mr,esc_term)
    end

    assert( c_alpha._strip )

    pattern_strip = c_alpha._strip

    local c =0 for i in pairs(b64d) do c=c+1 end

    assert( c_alpha._alpha == s,        "Integrity error." )
    assert( c == 64,                    "The alphabet must be 64 unique values." )
    if esc_term ~= "" then
      assert( not c_alpha._alpha:find(esc_term), "Tail characters must not exist in alphabet." )
    end

    if known_base64_alphabets[alpha] == nil then
      known_base64_alphabets[alpha]=c_alpha
    end
  end

  return c_alpha._alpha,c_alpha._term
end


--------------------------------------------------------------------------------
-- encode64
--
--  Entry point mode selector.
--
--
local function encode64(i,o)
  local method

  if o ~= nil and (io.type(o) == "file" or class.is_a(o,aprilio.stream)) then
    local file_out = o
    o = function(s) file_out:write(s) end
  end

  if type(i) == "string" then
    if type(o) == "function" then
      method = encode64_with_predicate
    else
      assert( o == nil, "unsupported request")
      method = encode64_tostring
    end
  elseif io.type(i) == "file" or class.is_a(i,aprilio.string) then
    assert( type(o) == "function", "file source requires output predicate")
    i      = encode64_io_iterator(i)
    method = encode64_with_ii
  else
    assert( false, "unsupported mode" )
  end

  return method(i,o)
end


--------------------------------------------------------------------------------
-- decode64
--
--  Entry point mode selector.
--
--
local function decode64(i,o)
  local method

  if o ~= nil and (io.type(o) == "file" or class.is_a(o,aprilio.stream)) then
    local file_out = o
    o = function(s) file_out:write(s) end
  end

  if type(i) == "string" then
    if type(o) == "function" then
      method = decode64_with_predicate
    else
      assert( o == nil, "unsupported request")
      method = decode64_tostring
    end
  elseif io.type(i) == "file" or class.is_a(i,aprilio.stream) then
    assert( type(o) == "function", "file source requires output predicate")
    i      = decode64_io_iterator(i)
    method = decode64_with_ii
  else
    assert( false, "unsupported mode" )
  end

  return method(i,o)
end

set_and_get_alphabet("base64")

--[[**************************************************************************]]
--[[******************************  Module  **********************************]]
--[[**************************************************************************]]
local ee5_base64 = {
  encode      = encode64,
  decode      = decode64,
  alpha       = set_and_get_alphabet,
}

package.loaded["ee5_base64"] = ee5_base64
_G.ee5_base64 = ee5_base64
return ee5_base64
