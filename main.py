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
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from collections import defaultdict

_WHISPER_MODEL_NAME = "turbo"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def transcribe_and_diarize(
    model: nn.Module,
    pipeline: pyannote.audio.Pipeline,
    audio_path: str,
) -> dict:
    print(f"Processing audio {audio_path}")
    diarization_result = pipeline(audio_path, num_speakers=2)
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
        result = model.transcribe(temp_audio_path, word_timestamps=True, language="en")
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


def get_unique_entities(text, analyzer_results):
    pii_map = {}
    pii_counter = defaultdict(int)
    for res in analyzer_results:
        span_text = text[res.start : res.end]
        ent = res.entity_type
        # Use span_text only as key; value can still encode entity type
        if span_text not in pii_map:
            pii_map[span_text] = f"{ent}_{pii_counter[ent]}"
            pii_counter[ent] += 1
    return pii_map


def anonymize_with_map(pii_map, text, entity_type=None):
    return pii_map.get(text, "UNKNOWN")


def transcript_deidentifcation(save_dir):
    print(f"Applying de-identification...")
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    for transcript in os.listdir(save_dir):
        with open(os.path.join(save_dir, transcript), "r") as f:
            content = f.read()

        analyzer_results = analyzer.analyze(
            text=content, language="en", allow_list=["SPEAKER_00", "SPEAKER_01"]
        )
        analyzer_results = [
            result for result in analyzer_results if result.entity_type != "DATE_TIME"
        ]
        pii_map = get_unique_entities(content, analyzer_results)
        anonymized_result = anonymizer.anonymize(
            text=content,
            analyzer_results=analyzer_results,
            operators={
                "DEFAULT": OperatorConfig(
                    "custom",
                    {
                        "lambda": lambda text, entity_type=None: anonymize_with_map(
                            pii_map, text, entity_type
                        )
                    },
                )
            },
        )

        with open(os.path.join(save_dir, transcript), "w", encoding="utf-8") as f:
            f.writelines(anonymized_result.text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token", help="Your HuggingFace Token to download models", required=True
    )
    parser.add_argument("--audio_dir", help="Directory with audio files", required=True)
    parser.add_argument(
        "--save_dir", help="Directory to save the transcripts", required=True
    )
    args = parser.parse_args()

    huggingface_hub.login(args.hf_token)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    audio_files = os.listdir(args.audio_dir)
    audio_files = [f for f in audio_files if f.endswith(".wav")]

    model = whisper.load_model(_WHISPER_MODEL_NAME, device=_DEVICE)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    ).to(torch.device(_DEVICE))

    for file in audio_files:
        print(file)
        full_transcript = transcribe_and_diarize(
            model=model,
            pipeline=pipeline,
            audio_path=os.path.join(args.audio_dir, file),
        )

        save_transcript(
            full_transcript=full_transcript,
            output_file_path=os.path.join(args.save_dir, f"{file.split('.')[0]}.txt"),
        )

    # Apply Deidentification to all transcripts in save_dir
    transcript_deidentifcation(args.save_dir)
