import argparse
from tqdm import tqdm
import glob
import os
from jamo import h2j
from g2pk import G2p
g2p = G2p()

def get_path(*args):
    return os.path.join('', *args)

def read_file(source_path):
    with open(source_path, mode="r", encoding="utf-8-sig", errors='ignore') as f:
        content = f.readline().rstrip()
    return content

def create_dictionary_final(source_path):
    phoneme_dict = {}
    for lab_file in tqdm(glob.glob(get_path(source_path, "*/*.lab"))):
        sentence = read_file(lab_file)
        for word in sentence.split(" "):
            if not word in phoneme_dict.keys():
                try:
                    phoneme_dict[word] = " ".join(h2j(g2p(word)))
                except:
                    print('fail')
                    print(word)
                    print(lab_file)
                    print(sentence)
    return phoneme_dict

def write_dictionary(savepath, dictionary):
    """
        input-dict format
            key: word of transcript delimited by <space> (e.g. 국물이)
            value: phoneme of hangul-word decomposed into syllables  (e.g. ㄱㅜㅇㅁㅜㄹㅣ)
                => i.e., input dictionary must define word-phoneme mapping
    """

    with open(savepath, "w", encoding="utf-8") as f:
        for key in dictionary.keys():
            content = "{}\t{}\n".format(key, dictionary[key])
            f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="/", help=".lab files path")
    parser.add_argument("--save_path", type=str, default="/", help="dictionary save path")
    parser.add_argument("--dict_file", type=str, default="dictionary.txt", help="dictionary file name")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    print("\n[LOG] create phoneme dictionary...")
    phoneme_dictionary = create_dictionary_final(args.in_path)

    print("\n[LOG] write grapheme and phoneme dictionary and metadata...")
    write_dictionary(savepath=os.path.join(args.save_path, args.dict_file), dictionary=phoneme_dictionary)
    print("[LOG] done!\n")
