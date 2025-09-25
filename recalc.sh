#!/bin/bash
prefix=$1
echo Prefix: $prefix

file1=$(wc -l $prefix*Reliance.jsonl 2>/dev/null | grep -v total | sort -g | tail -n 1 | awk '{ print $2 }')
file2=$(wc -l $prefix*Paraphrasability.jsonl 2>/dev/null | grep -v total | sort -g | tail -n 1 | awk '{ print $2 }')
file3=$(wc -l $prefix*Substitutability*.jsonl $prefix*Internalized*.jsonl 2>/dev/null | grep -v total | sort -g | tail -n 1 | awk '{ print $2 }')

echo F1: $file1
echo F2: $file2
echo F3: $file3

if [ -n "$file1" -a -n "$file2" -a -n "$file3" ]; then
    a=$(grep -v '"cot": ""' $file1 2>/dev/null | wc -l)
    b=$(grep -v '"cot": ""' $file2 2>/dev/null | wc -l)
    c=$(grep -v '"cot": ""' $file3 2>/dev/null | wc -l)
    min=$(echo -e "$a\n$b\n$c" | awk 'BEGIN {min = 9999999999} {if ($1 < min) min = $1} END {print min}')

    echo $a $b $c $min

    if [ $min -gt 10 ]; then
        echo -n "reliance "
        python src/print_organism_results.py --healthy $file1 --max-samples $min | tail -n 1 \
            | sed "s/'/\"/g" | jq -r '"\(.scores_stats.median | . * 1000 | round / 1000) (\(.scores_stats.p25 | . * 1000 | round / 1000), \(.scores_stats.p75 | . * 1000 | round / 1000))"'
        echo -n "paraph   "
        python src/print_organism_results.py --healthy $file2 --max-samples $min | tail -n 1 \
            | sed "s/'/\"/g" | jq -r '"\(.scores_stats.median | . * 1000 | round / 1000) (\(.scores_stats.p25 | . * 1000 | round / 1000), \(.scores_stats.p75 | . * 1000 | round / 1000))"'
        echo -n "substi   "
        python src/print_organism_results.py --healthy $file3 --max-samples $min | tail -n 1 \
            | sed "s/'/\"/g" | jq -r '"\(.scores_stats.median | . * 1000 | round / 1000) (\(.scores_stats.p25 | . * 1000 | round / 1000), \(.scores_stats.p75 | . * 1000 | round / 1000))"'
    fi
fi

