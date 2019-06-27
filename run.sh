#!/bin/bash
declare -arr graph=(../../datasets/dbpedia/dbpedia ../../datasets/author/author ../../datasets/enwiki/enwiki ../../datasets/friendster/friendster ../../datasets/LiveJournal/LJ ../../datasets/orkut/orkut ../../datasets/trackers/trackers ../../datasets/uspatent/patent ../../datasets/wiki/dbpedia_en ../../datasets/wiki-link/wiki-link)
declare -arr src=(1 2 1 1 1 1 1 1 1 1)
#declare -arr src=(0 1 0 0 0 0 0 0 0 0)

#declare -arr graph=(../dbpedia/dbpedia ../trackers/trackers ../enwiki/enwiki ../wiki/dbpedia_en ../wiki-link/wiki-link ../friendster/friendster)
#declare -arr graph=(../author/author)
#######declare -arr graph=(twitter/twitter_INT)
#declare -arr graph=(../friendster/friendster)
#declare -arr graph=(../large_LJ/lj large_orkut/orkut)
#declare -arr graph=(../roadCA/roadCA)

ptr=0
for file in ${graph[@]};do
#    file=../../datasets/trackers/trackers
#    file=../../datasets/orkut/orkut
    #    while[$src -lt 880] do
    #   for padding in 1 2 3 4 5
    #   do
    beg="$file"_beg_pos.bin
    csr="$file"_csr.bin
    csv="$file".csv
    #    echo
    #echo
    #echo
    #echo
    #echo ./bfs $beg $csr x 128 128 1 $csv 0.1 1 $file $padding
    #    ./bfs $beg $csr x 128 128 1 $csv 0.1 1 LJ 1
    mysrc=${src[$ptr]}

    echo $mysrc
    for (( loop = 0; loop < 5; loop++   ))
    do
        echo ./bfs $beg $csr x 128 128 1 $csv 0.1 $mysrc $file 1
 #        ./bfs $beg $csr x 128 128 0 $csv 0.1 $mysrc $file 1
       ./bfs $beg $csr x 256 256 1 $csv 0.1 $mysrc $file 1
    done
    #mv $file"_real_count_distribution.csv" .
    #sort -V $file"_real_count_distribution.csv" -o $file"_real_count_distribution_sort.csv"
    #mv $file"_real_count_distribution_sort.csv" .
    #cat LJ_real_count_distribution_sort.csv | uniq &>> orkut.csv
    #    let padding = padding+50
    #    done
    ptr=$((ptr+1))
done
#mv $file"_real_count_distribution.csv" .
#sort -V $file"_real_count_distribution.csv" -o $file"_real_count_distribution_sort.csv"
