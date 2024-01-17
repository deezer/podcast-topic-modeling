T=10

DIR=/Volumes/hyperion/exd
DATASET=../datasets/deezer_podcast_dataset.tsv

cd ptm/entity_linking
./download-files.sh ${DIR}
cd ../../

for i in 1 2
do
	cd ptm/entity_linking
	./launch-rel-batch.sh ${DIR} ../../../${DATASET} ${DIR} 64
	cd ../../

	python ptm/data_preprocessing/main_prepro.py --examples_file ${DATASET} \
	   --annotated_file ${DIR}/linked_entities.json \
	   --embeddings_file_path ${DIR}/enwiki_20180420_300d.pkl \
	   --path_to_save_results ${DIR}

	for K in 20 50 100 200
	do
		for alpha_ent in 0.30 0.40
		do

			python3 ptm/data_preprocessing/main_enrich_corpus_using_entities.py --prepro_file ${DIR}/prepro.txt \
			   --prepro_le_file ${DIR}/prepro_le.txt \
			   --vocab_file ${DIR}/vocab.txt \
			   --vocab_le_file ${DIR}/vocab_le.txt \
			   --embeddings_file_path ${DIR}/enwiki_20180420_300d.pkl \
			   --path_to_save_results ${DIR} \
			   --alpha_ent ${alpha_ent} \
			   --k ${K}

			for alpha_word in 0.2 0.3 0.4 0.5
			do

				python3 ptm/neice_model/main.py \
				--corpus ${DIR}/prepro_enrich_entities_th${alpha_ent}_k${K}.txt \
				--embeddings ${DIR}/enwiki_20180420_300d.pkl \
				--output_dir ${DIR} \
				--mask_entities_file ${DIR}/mask_enrich_entities_th${alpha_ent}_k${K}.npz \
				--vocab ${DIR}/new_vocab_th${alpha_ent}_k${K}.txt \
				--n_topics ${T} \
				--n_neighbours ${K} \
				--alpha_word ${alpha_word} \
				--NED

				cd ptm/evaluation
				echo "K ${K} T ${T} alpha_word ${alpha_word} alpha_ent ${alpha_ent}: " >> ../../results_deezer_${i}.txt
				./evaluate-topics.sh ${DIR}/top_words.txt ${T} ${DIR}/wikipedia_bd  >> ../../results_deezer_${i}.txt
				echo "\n" >> ../../results_deezer_${i}.txt
				cd ../../

			done
		done
	done
done
