import argparse
from utils import read_top_words_file

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Topics file")
    parser.add_argument("--output", type=str, required=True, help="Scores file")
    parser.add_argument("--T", type=int, default=10, help="Number of top words")
    args = parser.parse_args()

    topics = read_top_words_file(args.input)

    with open(args.output, 'w', encoding='utf-8') as f:
        for topic in topics:
            f.write(f'{" ".join(topic[:args.T])}\n')


if __name__ == "__main__": 
    main()