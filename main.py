import whisper
import torchaudio
import pyannote
from torch import nn
from pyannote.audio import Pipeline
from pyannote.core import Segment
from typing import List
import os
import torch
import huggingface_hub
import argparse

_WHISPER_MODEL_NAME = "large-v2"
_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def transcribe_and_diarize(
    model: nn.Module,
    pipeline: pyannote.audio.Pipeline,
    audio_path: str,
) -> dict:
    print(f"Processing audio {audio_path}")
    diarization_result = pipeline(audio_path)
    waveform, sample_rate = torchaudio.load(audio_path)

    full_transcript = []
    annotation_obj = diarization_result.speaker_diarization

    for segment, _, label in annotation_obj.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end

        # Calculate indices for slicing the audio tensor
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract the segment waveform
        segment_waveform = waveform[:, start_sample:end_sample]

        # Save the temporary segment to a file for Whisper processing
        temp_audio_path = "temp_segment.wav"
        torchaudio.save(temp_audio_path, segment_waveform, sample_rate)

        # Transcribe the segment using Whisper
        result = model.transcribe(temp_audio_path, word_timestamps=True)
        transcribed_text = result["text"].strip()

        # Remove the temporary file
        os.remove(temp_audio_path)
        if transcribed_text:
            full_transcript.append(
                {
                    "speaker": label,
                    "start": start_time,
                    "end": end_time,
                    "text": transcribed_text,
                }
            )

    return full_transcript


def save_transcript(full_transcript, output_file_path):
    file_output_lines = []
    for item in full_transcript:
        # Prepare console output (with timestamps)
        total_seconds = item["start"]
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        start_str = f"{minutes:02d}:{seconds:05.2f}"
        console_line = f"[{start_str}] {item['speaker']}: {item['text']}"
        print(console_line)

        # Prepare file output (without timestamps, as requested)
        file_line = f"{item['speaker']}: {item['text']}\n"
        file_output_lines.append(file_line)

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.writelines(file_output_lines)
        print(f"\nSuccessfully saved transcript to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_token", help="Your HuggingFace Token to download models")
    args = parser.parse_args()

    huggingface_hub.login(args.hf_token)

    model = whisper.load_model(_WHISPER_MODEL_NAME, device=_DEVICE)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    ).to(torch.device(_DEVICE))

    full_transcript = transcribe_and_diarize(
        model=model, pipeline=pipeline, audio_path="../data/harvard.wav"
    )

    save_transcript(
        full_transcript=full_transcript, output_file_path="../data/harvard.txt"
    )
