return {
  '"APRIL-ANN v" TOSTRING(APRILANN_VERSION_MAJOR) "." TOSTRING(APRILANN_VERSION_MINOR) "." TOSTRING(APRILANN_VERSION_RELEASE) " COMMIT " TOSTRING(GIT_COMMIT) " " TOSTRING(GIT_HASH)',
  '"Copyright (C) 2012-2015 DSIC-UPV, CEU-UCH"',
  string.format('"Compiled at %s, timestamp %u"', os.date(), os.time()),
  '"This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.\\nThis is free software, and you are welcome to redistribute it\\nunder certain conditions; see LICENSE.txt for details."',
}
