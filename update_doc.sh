#!/bin/bash
user=$(grep user /etc/pakozm.hol.es | cut -d' ' -f 2)
host=$(grep host /etc/pakozm.hol.es | cut -d' ' -f 2)
cmd="mirror -R doxygen_doc /public_html/STUFF/"
lftp -e "$cmd" -u $user $host
