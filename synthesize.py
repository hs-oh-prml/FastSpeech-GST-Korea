import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2pk import G2p
from jamo import h2j

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import symbol_to_sequence
from tqdm import tqdm
import os
from scipy.io import wavfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_korean(text):
    words = text.split(" ")
    g2p = G2p()
    phones = ""
    for w in words:
        ph = " ".join(h2j(g2p(w))) + " "
        phones += ph 

    phones = "{" + phones[:-1] + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(symbol_to_sequence(phones))

    return np.array(sequence)

def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    os.makedirs(train_config["path"]["result_path"], exist_ok=True)
    pitch_control, energy_control, duration_control = control_values

    for batch in tqdm(batchs):
        (ids, raw_texts, speakers,
         texts, text_lens, max_text_len,
         ref_mels, mel_lens, max_mel_len) = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                speakers=speakers,
                texts=texts,
                src_lens=text_lens,
                max_src_len=max_text_len,
                ref_mels=ref_mels,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

        mel = ref_mels.transpose(1, 2)
        lengths = mel_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]
        wav_predictions = vocoder_infer(
            mel, vocoder, model_config, preprocess_config, lengths=lengths
        )

        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        for wav, basename in zip(wav_predictions, ids):
            wavfile.write(os.path.join(
                train_config["path"]["result_path"],
                "{}_gt.wav".format(basename,)),
                          sampling_rate, wav)
        print("Generating Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=100000)
    parser.add_argument("--mode", type=str, choices=["batch", "single"], default="batch")
    parser.add_argument(
        "--source", type=str, default="/",
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id", type=int, default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p", "--preprocess_config", type=str, default="./config/preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default="./config/model.yaml",
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default="./config/train.yaml",
    )
    parser.add_argument(
        "--pitch_control", type=float, default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control", type=float, default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control", type=float, default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, "test.txt", preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        ref_mel = np.array([np.load(args.ref_mel)])
        texts = np.array([preprocess_korean(args.text)])
        text_lens = np.array([len(texts[0])])
        mel_lens = np.array([ref_mel[0].shape[0]])
        batchs = [(ids, raw_texts, speakers, 
        texts, text_lens, max(text_lens), 
        ref_mel, mel_lens, max(mel_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
