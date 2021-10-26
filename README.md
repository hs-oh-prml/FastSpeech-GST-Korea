# Korean FastSpeech 2 with GST - PyTorch Implementation
This project has only vanilla FastSpeech2 and GST. 

You should combine GST and FastSpeech2. Full code will be provided after 3 month.


![](./img/fastspeech2.png)

![](./img/gst.png)


## Requirements
- Python >= 3.7
- PyTorch >= 1.7.1
- librosa >= 0.8.0
- g2pk
- MFA

# Training

## Datasets
Korea Single Speaker Speech (KSS) Dataset [https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

Kaist Audiobook Dataset [https://aihub.or.kr/opendata/kaist-audiobook](https://aihub.or.kr/opendata/kaist-audiobook)

## Preprocessing
 
First, create '.lab' files. 

You can easily convert '.txt' file into '.lab' file by few code. 

Write the code according to your environment and prepare '.lab' files.

### Important
To runnign mfa, '.lab' files must be same directory as wav files.

Ex)
```python
f = open("[filename].txt", "r")
line = f.readline()
w = open("[filename].lab", "w")
w.write(line)
f.close()
w.close()
```

If you complete to prepare '.lab' files, run 'prepare_align.py' to create dictionary files.

```
python3 prepare_align.py --in_path [.lab files path] --save_path [dictionary save path]
```

### Montreal Forced Aligner (Anaconda environment)

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.

It is recommended to build a new virtual environment different from the virtual environment to be used for learning.

``` 
conda create -n alinger -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch 
conda activate aligner 
pip install montreal-forced-aligner 
mfa thirdparty download
```

#### Important 
To training mfa, you must have the following directory structure.

```bash
[data_path]
    └───[spk 1]
            ├───[utterence 1].lab
            ├───[utterence 1].wav
            ├───[utterence 2].lab
            ├───[utterence 2].wav
            ├─── ...        
    └───[spk 2]
            ├───[utterence 1].lab
            ├───[utterence 1].wav
            ├───[utterence 2].lab
            ├───[utterence 2].wav
            ├─── ...
    └─── ....
```

and Train mfa

```bash
mfa train [data_path] [dictionary_path] [output_path] –c
```

It is recommended to set [output_path] 'your_data_path/TextGrid'.

After that, run the preprocessing script by

```bash
python3 preprocess.py
```

you should check 'preprocessed_path' in './config/preprecess.yaml' before run 'preprocess.py' 

The outputs of 'preprocess.py' are mel-spectrograms, pitchs, energys, durations, train.txt, val.txt, stat.json, speaker.json

## Training

Train your model with

```bash
python3 train.py
```

if you want to change some training setting, then modify './config/train.yaml'

# TensorBoard

Use

```bash
tensorboard --logdir output/log/
```

![](./img/tensorboard.png)

![](./img/mel.png)


# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017), Y. Wang, *et al*.

- [ming024's FastSpeech implementation](https://github.com/ming024/FastSpeech2)
- [KinglittleQ's GST-Tacotron implementation](https://github.com/KinglittleQ/GST-Tacotron)
