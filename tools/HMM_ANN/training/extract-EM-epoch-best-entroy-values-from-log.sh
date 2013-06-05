#!/bin/bash
 grep -v "#" $1  |
 awk 'BEGIN{n=1;last=0;}{
  if ($2 != n) {
     printf("%d %f\n", n, last);
     n=$2;
  }
  else last = $12;
}'
