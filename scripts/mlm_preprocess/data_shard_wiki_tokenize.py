import argparse, json, random
import os, itertools
from pathlib import Path
import sentencepiece as spm
import multiprocessing
from functools import partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_line', type=int, default=10000)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='x')
    parser.add_argument('--spm', type=str)
    args = parser.parse_args()
    
    lines_per_file = args.num_line
    smallfile = None
    file_id = 0
    sp = spm.SentencePieceProcessor(model_file=args.spm)

    input_folder, lang, start_id = args.data.split('@@@')
    start_id = int(start_id)

    meta_file = os.path.join(input_folder, lang, 'metadata.{}.json'.format(lang))
    with open(meta_file, "r") as fm:
        meta_data = json.load(fm)
    print ("Tokenizing language {}: num_files = {}, chunk_size = {}, start_id = {}".format(lang, meta_data["num_files"], meta_data["chunk_size"], start_id))
    #step = 500
    step = meta_data["num_files"]
    for i in range(start_id, start_id + step):
        input_file = os.path.join(input_folder, lang, '{}.{}.txt'.format(lang, i))
        print("| file ", input_file, flush=True)

        if not os.path.exists(input_file):
            break

        output_file= os.path.join(args.output_folder, lang, '{}.{}.txt'.format(lang, i))
        Path(os.path.join(args.output_folder, lang)).mkdir(parents=True, exist_ok=True)

        with open(input_file) as bigfile:
            lines = bigfile.readlines()
            with multiprocessing.Pool(39) as pool:
                src_tokenized_tokens = pool.map(partial(sp.encode, out_type=str), lines)
                pool.close()
                pool.join()
        
        smallfile = open(output_file, "w", encoding="utf-8")
        for src_tokenized_token in src_tokenized_tokens:
            smallfile.write(' '.join(src_tokenized_token) + '\n')
        if smallfile:
            smallfile.close()

if __name__ == "__main__":
    main()
