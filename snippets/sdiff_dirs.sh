# cmp_dir - program to compare two directories
BASE_DIR=/usr/config_check
ARCHIVE_DIR=$BASE_DIR/archive
OUTPUT_DIR=$BASE_DIR/output

function Helps ()
{
  printf "\n"
  echo "Usage: ./filecomp.sh [-lL] [-wW] oldfiles_dir newfile_dir"
  echo "-l|-L: Line by Line file comparision check"
  echo "-w|-W: Word by Word file comparision check"
  echo
  echo "-Example-"
  echo "./filecomp.sh -l oldfile_dir newfile_dir or ./filecomp.sh -L oldfile_dir newfile_dir"
  echo "./filecomp.sh -w oldfile_dir newfile_dir  or ./filecomp.sh -W oldfile_dir newfile_dir"
  echo ""
  printf "\n"
  return 0
}
function filecomp () {
#if [ $# -ne 3 ]; then
#  echo "usage:./filecomp.sh -l old_dir new_dir" 1>&2; exit 1
#  echo "usage:./filecomp.sh -w old_dir new_dir" 1>&2; exit 1
#fi
if [ $1 == "L" ]; then
echo  "Starting Line by Line file comparision execution with opition -L"
#       if [ ! -d "$2" ]; then
#               echo "$2 is not a directory!" 1>&2; exit 1
#       fi
#       if [ ! -d "$3" ]; then
#               echo "$3 is not a directory!"  1>&2;  exit 1
#       fi
        rm -f resultfile_*.txt
        missing=0
        old_dir="$2" new_dir="$3"

        find "$old_dir" -name '*' -print > /tmp/old_paths.x;
                while read old_path; do
                        echo "old_path = $old_path" >> resultfile_$(date +%Y%m%d%H%M%S).txt
                        filename=`basename "$old_path"`
                        find "$new_dir" -name "$filename" -print > /tmp/new_path.x
                        count=`cat /tmp/new_path.x | wc -l`
                                if [ $count -eq 1 ]; then
                                        new_path=`cat /tmp/new_path.x`
                                        echo "Comparison with $new_path:" >> resultfile_$(date +%Y%m%d%H%M%S).t                                              xt
                                        #sort -o $old_path $old_path
                                        #sort -o $new_path $new_path
                                        sdiff -s $old_path $new_path >> resultfile_$(date +%Y%m%d%H%M%S).txt
                                elif [ $count -gt 1 ]; then
                                        echo "ERROR: $count found under $new_dir" >> resultfile_$(date +%Y%m%d%                                              H%M%S).txt
  else
                                        echo "ERROR: does not exist under $new_dir" >> resultfile_$(date +%Y%m%                                              d%H%M%S).txt
                                         missing=`expr $missing + 1`
                                fi
                        echo "------------------" >> resultfile_$(date +%Y%m%d%H%M%S).txt
                done < /tmp/old_paths.x
                        echo "Missing: $missing" >> resultfile_$(date +%Y%m%d%H%M%S).txt
                        echo "File comparision done, please see resultfile_$(date +%Y%m%d%H%M%S).txt"
                        RES_FILE=resultfile_$(date +%Y%m%d%H%M%S).txt
                        if [ -s $RES_FILE ]; then
                                echo -n "Press ENTER: "; read; vi $RES_FILE
                        else
                        echo "$RES_FILE is empty.."
                        fi
#elif [ $1 == "W" ]; then
#       echo "Word by Word comparsion"
#fi
fi
}

##########
## MAIN ##
##########
option=$@
arg=($option)
case ${arg[0]} in
-l|-L)
        echo  "Starting Line by Line file comparision execution with opition -L"
        filecomp L $2 $3
        ;;
-w|-W)
        echo "Starting word by word file comparision execution with opition -W"
        filecomp W $2 $3
        ;;
-h|?|*)
        Helps
        exit
        ;;
esac
