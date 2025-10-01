# Transcribe Audio and De-identify PHI Data
This codebase provides a simple script to transcribe raw audio to text using Whipser and diarization using Pyannote. Furthermore, identifiable information in the transcripts are replaced with entities using Presidio.  

## Pre-requisites 

#### HuggingFace
On huggingface perform the following tasks: 

1. Create a hugging face [token](https://huggingface.co/settings/tokens) with Read type. Save the token securely as it cannot be viewed again.
2. Apply for access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). You will need to enter your University, Website, and usage. You will receive instant access.
3. Apply for access to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
4. Apply for access to [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Installation 

#### Computer installations
1. Install ffmpeg on your machine. For Ubuntu or Linux based systems, run the following commands:
```
sudo apt-get update
sudo apt-get ffmpeg
```

For other OS, visit the [official website](https://www.ffmpeg.org/download.html).

2. Install miniconda using the following [instructions](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer).

#### Python environment installation

1. Create a conda environment: ```conda create -n whisper python=3.10```
2. Activate environment: ```conda activate whisper```
3. Install necessary packages: ```pip install -r requirements.txt``` 

## How to use

#### Convert audio files to .mp3
1. Go to ```convert.sh``` and change directory path in line 3 to the directory with audio files. 
2. Run ```convert.sh```

#### Running the main script 
1. The main script requires three command line arguments: (1) HuggingFace token, (2) Directory with audio files, and (3) Directory to save the transcripts. 

2. Run the main script:
```
python main.py --hf_token="<insert_your_hugginface_token>" --audio_dir="<path_to_audio_file>" --save_dir="<path_to_save_transcripts>"
```