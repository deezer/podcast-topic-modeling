#!/bin/bash

if [ "$#" -ne 4 ]
then
    echo 'Illegal number of parameters'
    echo './launch_rel_batch.sh $OUTPUT_DIR $DATASET_FNAME $ENTITY_LINKER_PATH_DIR $BATCH_SIZE'
    exit
fi

OUTPUT_DIR=$1
DATASET_FNAME=$2
BASE_URL=$3
WIKI_VERSION='wiki_2019'
BATCH_SIZE=$4

if [ ! -f ${DATASET_FNAME} ]
then
    echo "${DATASET_FNAME} does not exist!"
    exit
fi

mkdir -p ${OUTPUT_DIR}

python3 radboud_entity_linker_batch.py ${DATASET_FNAME} ${OUTPUT_DIR} ${BASE_URL} \
    --batch_size ${BATCH_SIZE} \
    --wiki_version ${WIKI_VERSION}

python3 join_predictions.py ${OUTPUT_DIR} ${DATASET_FNAME} \
    --batch_size ${BATCH_SIZE}
