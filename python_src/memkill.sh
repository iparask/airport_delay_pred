#!/bin/bash

THRESHOLD="90"

while true
do

  if [ `awk '/^Mem/ {printf("%u", 100*$3/$2);}' <(free -m)` -gt $THRESHOLD ]
  then
    ps aux | grep -i python | awk '{if($1=="iparask") print $2}' | while read -r pid ; do
        #to write the kill "$pid"
        echo "$pid"
        kill -9  "$pid"
    done
  fi
  sleep 5

done