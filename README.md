# Topic Modeling on Podcast Short-Text Metadata

The current repository provides the code for NEiCE, a Named Entity (NE) informed corpus embedding for short-text topic modeling with Non-negative Matrix Factorization (NMF). This is part of the experiments described in the article [**Topic Modeling on Podcast Short-Text Metadata**](https://arxiv.org/pdf/2201.04419.pdf), presented at [**ECIR 2022**](https://ecir2022.org/).

In order to reproduce the experiments concerning the other baselines, please refer to the following repositories:

- [CluWords](https://github.com/feliperviegas/cluwords)

- [GPU-DMM](https://github.com/WHUIR/GPUDMM)

- [NQTM](https://github.com/BobXWu/NQTM)

- [SeaNMF](https://github.com/tshi04/SeaNMF)

The current project consists of four parts:

-  `ptm/entity_linking`: the extraction of named entities from the podcast metadata and their linking to Wikipedia entities. See [Entity Linking](https://github.com/deezer/podcast-topic-modeling#entity-linking) for more details.

- `ptm/data_preprocessing`: the preprocessing of podcast metadata by taking into account the extracted named entities. See [Data Preprocessing](https://github.com/deezer/podcast-topic-modeling#data-Preprocessing) for more details.

- `ptm/neice_model`: the implementation of NEiCE. See [NEiCE Model](https://github.com/deezer/podcast-topic-modeling#neice-model) for more details.

- `ptm/evaluation`: the evaluation of the extracted topics in terms of topic coherence. See [Evaluation](https://github.com/deezer/podcast-topic-modeling#evaluation) for more details.

`ptm/data_preprocessing` and `ptm/evaluation` should be used with the other baselines too (CluWords, GPU-DMM, NQTM, SeaNMF).

## Installation

```bash
git clone git@github.com:deezer/podcast-topic-modeling.git
cd podcast-topic-modeling
```
The directory `docker` contains two Docker files. The first is the [`experiments`](https://github.com/deezer/podcast-topic-modeling/tree/main/docker/experiments/Dockerfile) image which should be used to run the experiment's code with a python environment. The second one is the [`evaluation`](https://github.com/deezer/podcast-topic-modeling/tree/main/docker/evaluation/Dockerfile) image which should be used to run the evaluation's code with a Java environment (compatible to run [palmetto](https://github.com/dice-group/Palmetto) locally).

An example of how to build and run the `experiments` image is provided below:

```bash
cd docker/experiments
docker build --tag experiments .
export DATA_PATH=<Your_Data_Path>
export CODE_PATH=<Your_Code_Path>
docker run --rm -ti --name=experiments --gpus device=0 --memory 16G -v $CODE_PATH/:/workspace -v $DATA_PATH:/data experiments /bin/bash
```

## Reproduce published results

We further explain how to reproduce the results reported in Table 5 of the article.

*NOTE: we updated the code to the latest Entity Linking packages and model versions ([REL](https://github.com/informagi/REL) and [flairNLP](https://github.com/flairNLP/flair)) as well as to the latest version of [names_dataset](https://pypi.org/project/names-dataset/). For this reason, the discovered named entities and word vocabulary are not exactly the same which leads to differences in topic coherence scores. Nevertheless, the scores follow the same trends and lead to the same conclusion as presented in the paper.*

### Download data
You will find on [Zenodo](https://zenodo.org/record/5834061#.Yd2ZaljMLlz) the **Deezer Podcast Dataset** (`deezer_podcast_dataset.tsv`) which was released and used in the paper. This dataset contains 29,539 shows and is composed of 2 columns (see section 4 of the article for more details). The first column corresponds to the podcast *titles*. The second column corresponds to the podcast *descriptions*.

If you want to run our code on the other datasets mentioned in the paper you can find them [here](https://github.com/odenizgiz/Podcasts-Data) (the **iTunes** dataset) and [here](https://podcastsdataset.byspotify.com/) (the **Spotify** dataset).

### Entity Linking

In this step, we extract and link named entities (NEs) from the podcast metadata as presented in **Preprocessing step** in section 3.2. of the article.
 
**Requirements**
- Build [`experiment`](https://github.com/deezer/podcast-topic-modeling/tree/main/docker/experiments/Dockerfile) Docker image as shown above.
- Download pre-trained Entity Linker models as below:
```
cd entity_linking/
./download-files.sh $OUTPUT_EL_DIR
```
`$OUTPUT_EL_DIR` is the directory where the pre-trained Entity Linker models are stored.
 
**Usage**

Generate a JSON file which contains all the NEs extracted and linked from the podcast metadata:
```
./launch-rel-batch.sh $OUTPUT_DIR $DATASET $ENTITY_LINKER_PATH_DIR $BATCH_SIZE
```
- `$OUTPUT_DIR`: the directory where to save the extracted NEs.
- `$DATASET`: the file path of the podcast metadata (e.g. `./dataset/deezer_podcast_dataset.tsv`).
- `$ENTITY_LINKER_PATH_DIR`: the directory where the pre-trained Entity Linker models are stored. Same value as `$OUTPUT_EL_DIR` in the previous step.
- `$BATCH_SIZE`: batch size (e.g., 512). Adjust this parameter to the computation capacity of which you dispose.
 
The output of this command is `$OUTPUT_DIR/linked_entities.json`.


### Data Preprocessing
 
Perform the preprocessing of the podcast metadata taking into account the NEs identified in the previous step (see **Preprocessing step** in section 3.2. of the article for more details).
 
**Requirements**:
- Build [`experiment`](https://github.com/deezer/podcast-topic-modeling/tree/main/docker/experiments/Dockerfile) Docker image as shown above.
- Download the pre-trained Wikipedia2Vec model:
```
cd ../data_preprocessing/
wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2 -P $SAVE_DIR
bzip2 -d $SAVE_DIR/enwiki_20180420_300d.pkl.bz2
```
`$SAVE_DIR` is the directory where to store the pre-trained Wikipedia2Vec model.
 
**Usage**
 
The preprocessing is carried out in two steps.

#### First step
Apply a basic preprocessing, namely:
1. Tokenize each text and remove stop words.
2. Filter out short texts with length less than 2 words.
3. Remove words with document frequency less than 5.
4. Convert all letters to lower cases, ignoring NEs whose confidence score is greater than 0.9.
 
```
python main_prepro.py --examples_file $DATASET \
   --annotated_file $LINKED_ENTITIES_JSON \
   --embeddings_file_path $WIKIPEDIA2VEC_EMBEDDINGS_FILE \
   --path_to_save_results $OUTPUT_DIR
```
Input:
- ` $DATASET`: the file path of the podcast metadata(`deezer_podcast_dataset.tsv`)
- ` $LINKED_ENTITIES_JSON`: the file `linked_entities.json` created for the concerned dataset during the [Entity Linking](https://github.com/deezer/podcast-topic-modeling#entity-linking) step.
- ` $WIKIPEDIA2VEC_EMBEDDINGS_FILE`: the Wikipedia2Vec file (`enwiki_20180420_300d.pkl`) downloaded in the requirements step.
- ` $OUTPUT_DIR`: the directory where to store the output.
 
Output:
- `$OUTPUT_DIR/prepro.txt`: the preprocessed dataset without identified NEs.
- `$OUTPUT_DIR/prepro_le.txt`: the preprocessed dataset with NEs.
- `$OUTPUT_DIR/vocab.txt`: the vocabulary of the preprocessed dataset with single words only.
- `$OUTPUT_DIR/vocab_le.txt`: the vocabulary of the preprocessed dataset with both single words and NEs.
 
 
#### Second step
Obtain the single words that are the most similar to the NEs extracted from each podcast title and description.
```
python main_enrich_corpus_using_entities.py --prepro_file $INPUT_DIR/prepro.txt \
   --prepro_le_file $INPUT_DIR/prepro_le.txt \
   --vocab_file $INPUT_DIR/vocab.txt \
   --vocab_le_file $INPUT_DIR/vocab_le.txt \
   --embeddings_file_path $WIKIPEDIA2VEC_EMBEDDINGS_FILE \
   --path_to_save_results $PATH_TO_SAVE_RESULTS \
   --alpha_ent 0.3 \
   --k 500
```
Input:
- `$INPUT_DIR`: the output directory of the previous step.
- `$WIKIPEDIA2VEC_EMBEDDINGS_FILE`: the Wikipedia2Vec file `enwiki_20180420_300d.pkl` downloaded in the requirements step.
- `$PATH_TO_SAVE_RESULTS`: the directory where to store the outputs.
- `alpha_ent`: minimum cosine similarity score between single words and entities.
- `k`: maximum number of nearest single words per entity.

Output:
- `mask_enrich_entities_th0.30_k500.npz`: a NE-related mask per podcast where the single words most similar to the NEs will have a value of 1 and the rest 0.
- `new_vocab_th0.30_k500.txt`: new vocabulary file without NEs.
- `prepro_enrich_entities_th0.30_k500.txt`: the extended preprocessed corpus where the NEs are replaced by similar single words.


### NEiCE Model
Apply NEiCE to the extended preprocessed podcast metadata corpus (see **Computation step** in section 3.2 of the article for more details).
 
**Requirements**:
- Build [`experiment`](https://github.com/deezer/podcast-topic-modeling/tree/main/docker/experiments/Dockerfile) Docker image as shown above.
- Download the pre-trained Wikipedia2Vec model (see requirements in [Data Preprocessing](https://github.com/deezer/podcast-topic-modeling#data-Preprocessing)).
 
**Usage**

Extract `K` topics from the podcast metadata corpus and for each topic its `T` tops words.
```
cd ../neice_model/
python main.py \
  --corpus $PATH_TO_SAVE_RESULTS/prepro_enrich_entities_th0.30_k500.txt \
  --embeddings $WIKIPEDIA2VEC_EMBEDDINGS_FILE \
  --output_dir $OUPUT_DIR \
  --mask_entities_file $PATH_TO_SAVE_RESULTS/mask_enrich_entities_th0.30_k500.npz \
  --vocab $PATH_TO_SAVE_RESULTS/new_vocab_th0.30_k500.txt \
  --n_topics 50 \
  --n_neighbours 500 \
  --alpha_word 0.4 \
  --NED
```
 
Input:
- `$PATH_TO_SAVE_RESULTS` is the same as in the previous step (Second step of Data Preprocessing)
- `n_topics`: number of topics to extract.
- `n_neighbours`: maximum number of neighbors per cluword.
- `alpha_word`: minimum cosine similarity score between single words to be considered neighbors in a cluword.
- `$NED`: NE promoting strategy that gives maximum weight to singles words that are similar to NEs.

Output:
- `$OUPUT_DIR/top_words.txt`: the file that contains the top words for each extracted topic.
- `$OUPUT_DIR/W.txt`: the file that contains document encodings [n_docs, n_topics].
- `$OUPUT_DIR/H.txt`: the file that contains topic encodings [n_topics, n_words].

### Evaluation
Compute CV to evaluate the coherence of the topics extracted with NEiCE (see section 5 of the article for more details).
 
**Requirements**:
- Build [`evaluation`](https://github.com/deezer/podcast-topic-modeling/tree/main/docker/evaluation/Dockerfile) Docker image as shown above.
- Download `Palmetto`:
```
cd evaluation/
wget --no-check-certificate https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar
```
- Download indices of word co-occurences in Wikipedia (`wikipedia_bd`):
```
wget --no-check-certificate https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip -P $SAVE_DIR
unzip $SAVE_DIR/Wikipedia_bd.zip
rm -rf $SAVE_DIR/Wikipedia_bd.zip
```
 
**Usage**

Print the CV scores:

```
./evaluate-topics.sh $TOPICS_MODEL_FILE $NUMBER_TOP_WORDS $WIKIPEDIA_DB_PATH
```
Input:
- `$TOPICS_MODEL_FILE`: the file which contains the top words of each topic (`top_words.txt`).
- `$NUMBER_TOP_WORDS`: the number of words considered per topic (`T`).
- `$WIKIPEDIA_DB_PATH`: the path to the Wikipedia database `wikipedia_bd`, previously downloaded and unzipped.

## Cite

Please cite our paper if you use this code in your work:

```BibTeX
@inproceedings{Valero2022,
  title={Topic Modeling on Podcast Short-Text Metadata},
  author={Valero, Francisco B. and Baranes, Marion and Epure, Elena V.},
  booktitle={44th European Conference on Information Retrieval (ECIR)},
  year={2022}
}
```
