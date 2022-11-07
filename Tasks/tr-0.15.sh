i=0
while(($i<5))
do
    start=`expr $i \* 512`
    # echo $start
    echo 起始位点为$start
    source delpt.sh
    python3 TaskForDNAMLM.py -mr 0.15 -is $start > rcd/m15/tr_$start.log
    i=`expr $i + 1`
done
