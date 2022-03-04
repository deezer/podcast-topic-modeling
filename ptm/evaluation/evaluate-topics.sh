#!/bin/bash

if [ $# -ne 3 ]
then 
    echo "Three parameters are required"
    echo './evaluate-topics.sh $TOPICS_MODEL_FILE $NUMBER_TOP_WORDS $WIKIPEDIA_DB_PATH'
    exit
fi
TMP_T_TOP_WORDS="/tmp/$2_top_words.txt"
python3 extract_t_top_words.py --input $1 --output ${TMP_T_TOP_WORDS} --T $2
TMP_CV_SCORES_RAW="/tmp/T=$2_CV_raw.txt"
java -jar palmetto-0.1.0-jar-with-dependencies.jar $3 "C_V" ${TMP_T_TOP_WORDS} > ${TMP_CV_SCORES_RAW}
TMP_CV_SCORES="/tmp/T=$2_CV.txt"
echo "CV" > ${TMP_CV_SCORES}
tail -n $2 ${TMP_CV_SCORES_RAW} | awk '{print $2}' >> ${TMP_CV_SCORES}
TMP_FILE=${TMP_CV_SCORES}
export TMP_FILE
python3 -c "import pandas as pd; import os; df = pd.read_csv(os.environ['TMP_FILE']); print(df['CV'].mean()* 100)"