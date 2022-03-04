import argparse
import datetime
import os
from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from utils import dataset_preprocessing, process_results, export_results
import pandas as pd
from REL.ner import  load_flair_ner


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_fname', type=str, help='Dataset filename')
    parser.add_argument('output_dir', type=str, help='Output path')
    parser.add_argument('base_url', type=str, help='REL data path')
    parser.add_argument('--wiki_version', type=str, default='wiki_2019', choices=['wiki_2014', 'wiki_2019'], help='REL Wiki version')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--col_title', type=str, default='title', help='Column name which contains the title')
    parser.add_argument('--col_description', type=str, default='description', help='Column name which contains the description')
    parser.add_argument('--sep', type=str, default='\t', help='Dataset separator')
    args = parser.parse_args()

    base_url = args.base_url
    wiki_version = args.wiki_version
    wiki_version_ = args.wiki_version.replace('_', '-')
    dataset_fname = args.dataset_fname
    output_dir = args.output_dir
    batch_size = args.batch_size
    config = {
        'mode': 'eval',
        'model_path': f'{base_url}/ed-{wiki_version_}/model',
    }
    print(f'{datetime.datetime.now()} - Reading dataset...')
    df = pd.read_csv(
        dataset_fname,
        sep=args.sep,
    )
    df[f'{args.col_title}_{args.col_description}'] = df[args.col_title] + " " + df[args.col_description]
    dataset = df[f'{args.col_title}_{args.col_description}'].to_list()
    del df
    num_examples = len(dataset)
    mention_detection = MentionDetection(base_url, wiki_version)
    tagger_ner = load_flair_ner("flair/ner-english-fast")
    model = EntityDisambiguation(base_url, wiki_version, config)
    for (batch, i) in enumerate(range(0, num_examples, batch_size)):
        j = min(i+batch_size, num_examples)
        input_text = dataset_preprocessing(dataset[i:j], e=i)
        print(f'{datetime.datetime.now()} - MentionDetection - {batch}...')
        mentions_dataset, _ = mention_detection.find_mentions(input_text, tagger_ner)
        print(f'{datetime.datetime.now()} - EntityDisambiguation - {batch} ...')
        predictions, _ = model.predict(mentions_dataset)
        result = process_results(mentions_dataset, predictions, input_text)
        le_fname = os.path.join(output_dir, f'linked_entities_{batch}.json')
        print(f'{datetime.datetime.now()} - Exporting results - {batch}...')
        export_results(le_fname, result)



if __name__ == "__main__": 
    main()