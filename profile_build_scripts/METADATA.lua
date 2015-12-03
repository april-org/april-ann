return {
  description = "A Pattern Recognizer In Lua",
  version = "0.4.1",
  prefix = os.getenv("PREFIX") or "/usr",
  url = "https://github.com/pakozm/april-ann",
  version_flags = {
    '-DAPRILANN_VERSION_MAJOR=0',
    '-DAPRILANN_VERSION_MINOR=4',
    '-DAPRILANN_VERSION_RELEASE=1',
  },
  disclaimer_strings =  {
    '"APRIL-ANN v" TOSTRING(APRILANN_VERSION_MAJOR) "." TOSTRING(APRILANN_VERSION_MINOR) "." TOSTRING(APRILANN_VERSION_RELEASE) "  Copyright (C) 2012-2015 DSIC-UPV, CEU-UCH"',
    string.format('"Compiled at %s, timestamp %u"', os.date(), os.time()),
    '"This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.\\nThis is free software, and you are welcome to redistribute it\\nunder certain conditions; see LICENSE.txt for details."',
  },
}
