#!/bin/bash

if [ $# -ne 1 ] 
then 
    echo "One parameter is required"
    echo './download-files.sh $OUTPUT_DIR'
    exit
fi

DIR_COMPRESSED_FILES="$1/COMPRESSED_FILES"

FILES=("generic")
FILES+=("ed-wiki-2019")
FILES+=("wiki_2019")

for FILE in ${FILES[@]}
do
	wget -P ${DIR_COMPRESSED_FILES} "http://gem.cs.ru.nl/${FILE}.tar.gz"
	tar xvzf "${DIR_COMPRESSED_FILES}/${FILE}.tar.gz" -C $1
done

rm -rf ${DIR_COMPRESSED_FILES}
