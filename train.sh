#!/bin/sh

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

helpFunction()
{
   echo "TODO"
   exit 1
}

while getopts "d:a:m:l:n:c:p: " opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      a ) a="$OPTARG" ;;
      m ) m="$OPTARG" ;;
      l ) l="$OPTARG" ;;
      n ) n="$OPTARG" ;;
      c ) c="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$a" ] || [ -z "$d" ] || [ -z "$m" ] || [ -z "$l" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi
d=$(get_abs_filename "$d")
a=$(get_abs_filename "$a")
l=$(get_abs_filename "$l")
if [ -z "$p" ]
then
 p=""
else
  p=$(get_abs_filename "$p")
fi
export CUDA_VISIBLE_DEVICES="$c"
cd ./train/tasks/semantic;  ./train.py -d "$d"  -ac "$a" -m "$m" -l "$l" -n "$n" -p "$p"