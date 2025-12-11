import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import random
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

class SlakhJamDataset(Dataset):
    def __init__(self, csv_path, stems_dir, processor, beats_context=5, beats_target=1, sample_rate=32000, split="train", target_role=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV (slakh_train.csv or slakh_val.csv).
            stems_dir (str): Root directory containing the stems (e.g., DATA/STEMS/SLAKH).
            processor (MusicgenProcessor): Hugging Face processor for audio tokenization.
            beats_context (int): Number of beats to use as input context.
            beats_target (int): Number of beats to predict.
            sample_rate (int): Target sample rate for MusicGen (usually 32000).
            split (str): "train" or "val" (useful for debug/logging).
            target_role (str, optional): If set, only train on stems with this role (e.g., "Drums").
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['bpm', 'track_name', 'stem_id', 'role'])
        
        self.stems_dir = stems_dir
        self.processor = processor
        self.beats_context = beats_context
        self.beats_target = beats_target
        self.sample_rate = sample_rate
        self.split = split
        self.target_role = target_role

        self.tracks = self.df.groupby('track_name')
        
        if self.target_role:
             valid_tracks = []
             for track_name, group in self.tracks:
                 if self.target_role in group['role'].values:
                     valid_tracks.append(track_name)
             self.track_names = valid_tracks
             print(f"[{split}] Filtered dataset for role '{target_role}': {len(self.track_names)} tracks found.")
        else:
             self.track_names = list(self.tracks.groups.keys())

    def __len__(self):
        return len(self.track_names)

    def _load_audio(self, path, offset_sec, duration_sec):
        """Loads a specific segment of audio."""
        try:
            info = torchaudio.info(path)
            sr = info.sample_rate
            
            frame_offset = int(offset_sec * sr)
            num_frames = int(duration_sec * sr)
            
            waveform, _ = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            return waveform
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(1, int(duration_sec * self.sample_rate))

    def __getitem__(self, idx):
        track_name = self.track_names[idx]
        track_data = self.tracks.get_group(track_name)
        
        try:
            bpm = float(track_data.iloc[0]['bpm'])
        except (ValueError, TypeError):
            bpm = 0.0
            
        if bpm < 90 or bpm > 250:
            return self.__getitem__((idx + 1) % len(self))
            
        beat_duration = 60.0 / bpm
        context_duration = self.beats_context * beat_duration
        target_duration = self.beats_target * beat_duration
        total_duration = context_duration + target_duration

        if self.target_role:
            candidates = track_data[track_data['role'] == self.target_role]
            if len(candidates) == 0:
                return self.__getitem__((idx + 1) % len(self))
            target_row = candidates.sample(1).iloc[0]
        else:
            valid_roles = ['Drums', 'Bass', 'Lead', 'Harmony']
            present_roles = [r for r in track_data['role'].unique() if r in valid_roles]
            
            if not present_roles:
                return self.__getitem__((idx + 1) % len(self))
            
            selected_role = random.choice(present_roles)
            
            candidates = track_data[track_data['role'] == selected_role]
            target_row = candidates.sample(1).iloc[0]
            
        target_stem_id = target_row['stem_id']
        target_role = target_row['role']
        
        other_stems = track_data[track_data['stem_id'] != target_stem_id]
        if len(other_stems) > 0:
            num_bg = random.randint(0, len(other_stems))
            bg_rows = other_stems.sample(num_bg)
        else:
            bg_rows = pd.DataFrame()

        target_path = os.path.join(self.stems_dir, track_name, f"{target_stem_id}.mp3")
        if not os.path.exists(target_path):
             return self.__getitem__((idx + 1) % len(self))

        try:
            full_target = self._load_audio(target_path, 0, 10000) 
        except:
             return self.__getitem__((idx + 1) % len(self))

        if full_target.pow(2).mean().sqrt() < 0.01:
             return self.__getitem__((idx + 1) % len(self))

        track_duration_sec = full_target.shape[1] / self.sample_rate

        if track_duration_sec < total_duration:
             return self.__getitem__((idx + 1) % len(self))

        total_available_beats = int(track_duration_sec / beat_duration)
        required_beats = self.beats_context + self.beats_target
        
        if total_available_beats <= required_beats:
             start_time = 0.0
             target_waveform = full_target[:, :int(total_duration * self.sample_rate)]
        else:
            max_start_beat = total_available_beats - required_beats
            
            force_active = (random.random() < 0.98)
            
            found_chunk = False
            for attempt in range(50):
                start_beat = random.randint(0, max_start_beat)
                start_time = start_beat * beat_duration
                
                start_sample = int(start_time * self.sample_rate)
                end_sample = start_sample + int(total_duration * self.sample_rate)
                
                if end_sample > full_target.shape[1]: continue
                
                chunk = full_target[:, start_sample:end_sample]
                
                if not force_active:
                    target_waveform = chunk
                    found_chunk = True
                    break
                
                split_point = int(context_duration * self.sample_rate)
                target_future = chunk[:, split_point:]
                
                if target_future.pow(2).mean().sqrt() > 0.02:
                    target_waveform = chunk
                    found_chunk = True
                    break
            
            if not found_chunk:
                return self.__getitem__((idx + 1) % len(self))
        
        split_point = int(context_duration * self.sample_rate)
        target_past = target_waveform[:, :split_point]
        target_future = target_waveform[:, split_point:]

        bg_mix = torch.zeros_like(target_past)
        
        for _, row in bg_rows.iterrows():
            stem_path = os.path.join(self.stems_dir, track_name, f"{row['stem_id']}.mp3")
            if os.path.exists(stem_path):
                bg_stem = self._load_audio(stem_path, start_time, context_duration)
                if bg_stem.shape[1] != bg_mix.shape[1]:
                    min_len = min(bg_stem.shape[1], bg_mix.shape[1])
                    bg_stem = bg_stem[:, :min_len]
                    bg_mix = bg_mix[:, :min_len]
                bg_mix += bg_stem

        input_mix = bg_mix + target_past
        
        if self.split == "train":
            n_steps = random.randint(-5, 5)
            if n_steps != 0:
                effects = [['pitch', str(n_steps * 100)]]
                
                try:
                    input_mix, _ = torchaudio.sox_effects.apply_effects_tensor(
                        input_mix, self.sample_rate, effects
                    )
                    
                    target_future, _ = torchaudio.sox_effects.apply_effects_tensor(
                        target_future, self.sample_rate, effects
                    )
                except Exception as e:
                    pass

        max_val = torch.max(torch.abs(input_mix))
        if max_val > 1.0:
            input_mix = input_mix / max_val
        
        return {
            "audio_prompt": input_mix,
            "audio_target": target_future,
            "bpm": bpm,
            "role": target_role
        }

def collate_fn(batch):
    
    audio_prompts = [item['audio_prompt'].squeeze(0).numpy() for item in batch]
    audio_targets = [item['audio_target'].squeeze(0).numpy() for item in batch]
    
    return {
        "audio_prompts": audio_prompts,
        "audio_targets": audio_targets,
        "bpms": [item['bpm'] for item in batch],
        "roles": [item['role'] for item in batch]
    }
