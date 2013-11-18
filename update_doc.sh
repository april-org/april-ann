#!/bin/bash
rsync -vrhz --rsh="ssh -l pako" doxygen_doc cafre.dsic.upv.es:~/public_html/STUFF
