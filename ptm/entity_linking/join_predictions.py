import argparse
import os
import pandas as pd
from utils import import_results, export_results


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory to load and to export predictions')
    parser.add_argument('fname_dataset', type=str, help='Filename dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--map_keys', action='store_true', help='Map keys of the partitions')
    parser.add_argument('--col_title', type=str, default='title', help='Column name which contains the title')
    parser.add_argument('--col_description', type=str, default='description', help='Column name which contains the description')
    parser.add_argument('--sep', type=str, default='\t', help='Dataset separator')
    args = parser.parse_args()

    all_res = dict()
    df = pd.read_csv(
        args.fname_dataset,
        sep=args.sep,
    )
    df[f'{args.col_title}_{args.col_description}'] = df[args.col_title] + " " + df[args.col_description]
    dataset = df[f'{args.col_title}_{args.col_description}'].to_list()
    del df

    if args.batch_size == -1:
        cfile = os.path.join(args.data_dir, f'linked_entities.json')
        all_res = import_results(cfile)

    else:
        num_examples = len(dataset)
        for (batch, i) in enumerate(range(0, num_examples, args.batch_size)):
            j = min(i+args.batch_size, num_examples)
            d = { str(key): str(value) for (key, value) in enumerate(range(i,j, 1))}
            cfile = os.path.join(args.data_dir, f'linked_entities_{batch}.json')
            c_res = import_results(cfile)
            if args.map_keys:
                c_res_mapped = { d[k]:v for (k,v) in c_res.items() }
                all_res = {**all_res, **c_res_mapped}
            else:
                all_res = {**all_res, **c_res}
        export_results(os.path.join(args.data_dir, f'linked_entities.json'), all_res)
    # ensure detections
    t = 0
    for (id_doc, linked_entities) in all_res.items():
        doc = dataset[int(id_doc)].rstrip()
        for linked_entity in linked_entities:
            b, e, entity, *_ = linked_entity
            assert entity == doc[b:b+e]
            t += 1
    print(f'{t} ent. -> {len(all_res)} docs. -> {len(dataset)} all docs.')


if __name__ == "__main__": 
    main()