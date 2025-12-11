import torch
import torchaudio
import pandas as pd
import random
import os
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel, PeftConfig

MODEL_PATH = "models/musicgen-small"
EVAL_ROLE = "Drums" # Set to role you want to evaluate. If checkpoint was trained for a specific role, set this accordingly.
CHECKPOINT_DIR = f"output_jam_lora_all" # Change if you have a different checkpoint directory
CHECKPOINT_TYPE = "best"
CHECKPOINT_EPOCH = 77 # Set to specific checkpoint epoch you wish to evaluate. Set to None to use the checkpoint with the best eval loss (not necessarilly the best-sounding!).
DATA_DIR = "DATA/STEMS/SLAKH"
VAL_CSV = "slakh_val.csv"
OUTPUT_DIR = f"eval_output_lora_{EVAL_ROLE}"
CONTEXT_BEATS = 20
GENERATE_BEATS = 20
BEATS_PER_INFERENCE = 1
SAMPLE_RATE = 32000

def load_audio(path, offset_sec, duration_sec):
    """Loads a specific segment of audio."""
    try:
        info = torchaudio.info(path)
        sr = info.sample_rate
        frame_offset = int(offset_sec * sr)
        num_frames = int(duration_sec * sr)
        
        if frame_offset + num_frames > info.num_frames:
            num_frames = info.num_frames - frame_offset
            
        waveform, _ = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        target_samples = int(duration_sec * SAMPLE_RATE)
        if waveform.shape[1] < target_samples:
            pad_len = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
        return waveform
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return torch.zeros(1, int(duration_sec * SAMPLE_RATE))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    
    model_to_load = MODEL_PATH
    
    best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best")
    latest_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest")
    
    if CHECKPOINT_EPOCH is not None:
        epoch_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{CHECKPOINT_EPOCH}")
        if os.path.exists(epoch_path):
            print(f"SUCCESS: Found checkpoint for epoch {CHECKPOINT_EPOCH} at {epoch_path}")
            model_to_load = epoch_path
        else:
            print(f"WARNING: Checkpoint for epoch {CHECKPOINT_EPOCH} not found at {epoch_path}")
            print("Falling back to base model.")
            
    elif CHECKPOINT_TYPE == "best":
        if os.path.exists(best_path):
            print(f"SUCCESS: Found 'checkpoint_best' at {best_path}")
            print("Loading the BEST model for evaluation...")
            model_to_load = best_path
        elif os.path.exists(latest_path):
            print(f"WARNING: 'checkpoint_best' not found. Falling back to 'checkpoint_latest' at {latest_path}")
            model_to_load = latest_path
        else:
            print(f"WARNING: No checkpoints found in {CHECKPOINT_DIR}. Using base model.")
            
    elif CHECKPOINT_TYPE == "latest":
        if os.path.exists(latest_path):
            print(f"SUCCESS: Found 'checkpoint_latest' at {latest_path}")
            print("Loading the LATEST model for evaluation...")
            model_to_load = latest_path
        elif os.path.exists(best_path):
            print(f"WARNING: 'checkpoint_latest' not found. Falling back to 'checkpoint_best' at {best_path}")
            model_to_load = best_path
        else:
            print(f"WARNING: No checkpoints found in {CHECKPOINT_DIR}. Using base model.")
            
    if model_to_load == MODEL_PATH and os.path.exists(CHECKPOINT_DIR):
        subdirs = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint") and d not in ["checkpoint_best", "checkpoint_latest"]]
        if subdirs:
            subdirs.sort(key=lambda x: int(x.split("-")[1]) if "-" in x else 0)
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, subdirs[-1])
            print(f"WARNING: No standard checkpoints found. Using numerically latest: {latest_checkpoint}")
            model_to_load = latest_checkpoint
    
    print(f"Loading from {model_to_load}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_PATH)
    
    if model_to_load != MODEL_PATH:
        print(f"Loading LoRA adapter from {model_to_load}...")
        model = PeftModel.from_pretrained(model, model_to_load)
        model = model.merge_and_unload()
    
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.bos_token_id if model.config.bos_token_id is not None else 2048
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = 2048

    if model.config.decoder.decoder_start_token_id is None:
        model.config.decoder.decoder_start_token_id = model.config.decoder_start_token_id
    
    if model.config.decoder.pad_token_id is None:
        model.config.decoder.pad_token_id = model.config.pad_token_id

    if model.generation_config.decoder_start_token_id is None:
        model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id

    model.to(device)
    model.eval()

    print("Selecting random track...")
    
    val_csv_path = VAL_CSV
    if os.path.exists("slakh_val_reclassified.csv"):
        print("Found slakh_val_reclassified.csv, using it for evaluation.")
        val_csv_path = "slakh_val_reclassified.csv"
    elif os.path.exists("slakh_val_audio_smart.csv"):
        print("Found slakh_val_audio_smart.csv, using it for evaluation.")
        val_csv_path = "slakh_val_audio_smart.csv"
        
    df = pd.read_csv(val_csv_path)
    df = df.dropna(subset=['bpm', 'track_name', 'stem_id', 'role'])
    
    if EVAL_ROLE:
        tracks_with_role = df[df['role'] == EVAL_ROLE]['track_name'].unique()
        if len(tracks_with_role) == 0:
            print(f"No tracks found with role {EVAL_ROLE}")
            return
        tracks = list(tracks_with_role)
        print(f"Filtered to {len(tracks)} tracks containing {EVAL_ROLE}")
    else:
        tracks = list(df['track_name'].unique())
    
    track_name = random.choice(tracks)
    track_data = df[df['track_name'] == track_name]
    bpm = track_data.iloc[0]['bpm']
    print(f"Selected Track: {track_name}, BPM: {bpm}")

    if EVAL_ROLE:
        target_row = track_data[track_data['role'] == EVAL_ROLE].sample(1).iloc[0]
    else:
        valid_roles = ['Drums', 'Bass', 'Lead', 'Harmony']
        valid_rows = track_data[track_data['role'].isin(valid_roles)]
        if len(valid_rows) == 0:
             print("No valid roles found in this track. Retrying...")
             return main()
        target_row = valid_rows.sample(1).iloc[0]
        
    target_stem_id = target_row['stem_id']
    target_role = target_row['role']
    print(f"Target Role: {target_role} ({target_stem_id})")
    
    role_prompts_map = {
        "Lead": "Lead.",
        "Drums": "Drums.",
        "Bass": "Bass.",
        "Harmony": "Accompaniment."
    }
    text_prompt = role_prompts_map.get(target_role, "")
    print(f"Using Text Prompt: '{text_prompt}'")

    other_stems = track_data[track_data['stem_id'] != target_stem_id]
    if len(other_stems) > 0:
        num_bg = random.randint(1, len(other_stems))
        bg_rows = other_stems.sample(num_bg)
    else:
        bg_rows = pd.DataFrame()
    print(f"Background Stems: {len(bg_rows)}")

    beat_duration = 60.0 / bpm
    total_duration = (CONTEXT_BEATS + GENERATE_BEATS) * beat_duration
    
    target_path = os.path.join(DATA_DIR, track_name, f"{target_stem_id}.mp3")
    try:
        info = torchaudio.info(target_path)
        track_len_sec = info.num_frames / info.sample_rate
    except:
        print("Error reading target file info. Exiting.")
        return

    if track_len_sec < total_duration:
        print("Track too short. Please run again.")
        return

    total_available_beats = int(track_len_sec / beat_duration)
    required_beats = CONTEXT_BEATS + GENERATE_BEATS
    
    if total_available_beats <= required_beats:
        start_time = 0.0
        start_beat = 0
    else:
        max_start_beat = total_available_beats - required_beats
        
        found_active = False
        for _ in range(10):
            start_beat = random.randint(0, max_start_beat)
            start_time = start_beat * beat_duration
            
            check_duration = GENERATE_BEATS * beat_duration
            check_offset = start_time + (CONTEXT_BEATS * beat_duration)
            
            if check_offset + check_duration > track_len_sec:
                continue
                
            target_check = load_audio(target_path, check_offset, check_duration)
            if target_check.pow(2).mean().sqrt() > 0.01:
                found_active = True
                break
        
        if not found_active:
            print("Could not find active target section in this track. Skipping...")
            return main() 
        
    print(f"Start Time: {start_time:.2f}s (Beat {start_beat})")

    print("Loading audio stems...")
    target_samples = int(total_duration * SAMPLE_RATE)
    bg_mix = torch.zeros(1, target_samples)
    
    for _, row in bg_rows.iterrows():
        stem_path = os.path.join(DATA_DIR, track_name, f"{row['stem_id']}.mp3")
        stem_wav = load_audio(stem_path, start_time, total_duration)
        
        if stem_wav.shape[1] > target_samples:
            stem_wav = stem_wav[:, :target_samples]
        elif stem_wav.shape[1] < target_samples:
            stem_wav = torch.nn.functional.pad(stem_wav, (0, target_samples - stem_wav.shape[1]))
            
        bg_mix += stem_wav
    
    context_duration = CONTEXT_BEATS * beat_duration
    target_context = load_audio(target_path, start_time, context_duration)

    print("Saving context...")
    
    full_context_mix = bg_mix.clone()
    
    target_len = target_context.shape[1]
    if target_len <= full_context_mix.shape[1]:
        full_context_mix[:, :target_len] += target_context
    else:
        full_context_mix += target_context[:, :full_context_mix.shape[1]]
    
    max_val = torch.max(torch.abs(full_context_mix))
    if max_val > 1.0:
        full_context_mix = full_context_mix / max_val
        
    out_path_context = os.path.join(OUTPUT_DIR, f"jam_{track_name}_{target_role}_context.wav")
    torchaudio.save(out_path_context, full_context_mix.cpu(), SAMPLE_RATE)

    current_target = target_context.to(device)
    bg_mix = bg_mix.to(device)

    full_target_audio = current_target.clone()

    print(f"Starting generation for {GENERATE_BEATS} beats ({BEATS_PER_INFERENCE} beats per inference)...")
    
    tokens_per_inference = int((60.0 / bpm) * 50 * BEATS_PER_INFERENCE) + 10
    
    for i in range(0, GENERATE_BEATS, BEATS_PER_INFERENCE):
        window_start_sec = i * beat_duration
        window_end_sec = (i + CONTEXT_BEATS) * beat_duration
        
        window_start_sample = int(window_start_sec * SAMPLE_RATE)
        window_end_sample = int(window_end_sec * SAMPLE_RATE)
        
        if window_end_sample > bg_mix.shape[1]:
            pad = window_end_sample - bg_mix.shape[1]
            bg_segment = torch.nn.functional.pad(bg_mix, (0, pad))[:, window_start_sample:]
        else:
            bg_segment = bg_mix[:, window_start_sample:window_end_sample]
        
        context_samples = int(CONTEXT_BEATS * beat_duration * SAMPLE_RATE)
        if full_target_audio.shape[1] < context_samples:
             target_segment = full_target_audio
        else:
             target_segment = full_target_audio[:, -context_samples:]
        
        min_len = min(bg_segment.shape[1], target_segment.shape[1])
        bg_segment = bg_segment[:, :min_len]
        target_segment = target_segment[:, :min_len]
        
        input_mix = bg_segment + target_segment
        
        max_val = torch.max(torch.abs(input_mix))
        if max_val > 1.0:
            input_mix = input_mix / max_val
            
        bg_input = bg_segment.squeeze(0).cpu()
        bg_inputs = processor(
            audio=bg_input,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            if bg_inputs["input_values"].dim() == 2:
                bg_inputs["input_values"] = bg_inputs["input_values"].unsqueeze(1)
            
            audio_encoder = model.audio_encoder
            prompt_outputs = audio_encoder.encode(bg_inputs["input_values"])
            prompt_codes = prompt_outputs.audio_codes.squeeze(0)
            
            embed_tokens = model.decoder.model.decoder.embed_tokens
            
            prompt_embeds = torch.zeros(
                prompt_codes.shape[0], 
                prompt_codes.shape[2], 
                model.decoder.config.hidden_size, 
                device=device
            )
            for c in range(model.decoder.num_codebooks):
                prompt_embeds += embed_tokens[c](prompt_codes[:, c, :])
            
            all_text_inputs = processor(
                text=[text_prompt, ""],
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            all_text_outputs = model.text_encoder(
                input_ids=all_text_inputs["input_ids"],
                attention_mask=all_text_inputs["attention_mask"]
            )
            all_text_embeds = all_text_outputs.last_hidden_state
            
            all_text_embeds = model.enc_to_dec_proj(all_text_embeds)
            
            text_embeds = all_text_embeds[0:1]
            null_text_embeds = all_text_embeds[1:2]
            
            text_attention_mask = all_text_inputs["attention_mask"][0:1]
            null_text_attention_mask = all_text_inputs["attention_mask"][1:2]
            
            prompt_mask = torch.ones(prompt_embeds.shape[:2], device=device)
            encoder_hidden_states = torch.cat([text_embeds, prompt_embeds], dim=1)
            encoder_attention_mask = torch.cat([text_attention_mask, prompt_mask], dim=1)
            
            null_encoder_hidden_states = torch.cat([null_text_embeds, prompt_embeds], dim=1)
            null_encoder_attention_mask = torch.cat([null_text_attention_mask, prompt_mask], dim=1)
            
            target_input = target_segment.squeeze(0).cpu()
            target_inputs = processor(
                audio=target_input,
                sampling_rate=SAMPLE_RATE,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            if target_inputs["input_values"].dim() == 2:
                target_inputs["input_values"] = target_inputs["input_values"].unsqueeze(1)
                
            target_outputs = audio_encoder.encode(target_inputs["input_values"])
            target_codes = target_outputs.audio_codes.squeeze(0)
            
            pad_id = model.decoder.config.pad_token_id
            num_codebooks = target_codes.shape[1]
            seq_len = target_codes.shape[2]
            max_len = seq_len + num_codebooks
            
            target_flat = target_codes.reshape(-1, seq_len)
            delayed_target, _ = model.decoder.build_delay_pattern_mask(
                target_flat, 
                pad_token_id=pad_id,
                max_length=max_len
            )
            delayed_target = delayed_target.reshape(target_codes.shape[0], num_codebooks, -1)
            
            decoder_start_token_id = model.decoder.config.decoder_start_token_id
            decoder_input_ids = delayed_target.new_zeros(delayed_target.shape)
            decoder_input_ids[:, :, 1:] = delayed_target[:, :, :-1].clone()
            decoder_input_ids[:, :, 0] = decoder_start_token_id
            
            original_proj = model.enc_to_dec_proj
            model.enc_to_dec_proj = torch.nn.Identity()
            
            guidance_scale = 3.0
            
            batch_encoder_hidden_states = torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
            batch_encoder_attention_mask = torch.cat([encoder_attention_mask, null_encoder_attention_mask], dim=0)
            
            current_input_ids = decoder_input_ids
            
            for step in range(tokens_per_inference):
                batch_input_ids = torch.cat([current_input_ids, current_input_ids], dim=0)
                
                outputs = model(
                    decoder_input_ids=batch_input_ids,
                    encoder_outputs=(batch_encoder_hidden_states,),
                    attention_mask=batch_encoder_attention_mask
                )
                
                logits = outputs.logits 
                
                cond_logits, uncond_logits = logits.chunk(2, dim=0)
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                
                if logits.dim() == 3:
                    logits = logits.reshape(current_input_ids.shape[0], num_codebooks, -1, logits.shape[-1])
                
                next_token_logits = logits[:, :, -1, :]

                if next_token_logits.shape[-1] > 2048:
                    next_token_logits[:, :, 2048] = -float('inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                next_tokens = next_tokens.view(current_input_ids.shape[0], num_codebooks, 1)
                
                current_input_ids = torch.cat([current_input_ids, next_tokens], dim=2)
            
            model.enc_to_dec_proj = original_proj
            
            gen_codes_list = []
            total_len = current_input_ids.shape[2]
            
            for c in range(num_codebooks):
                stream = current_input_ids[0, c, c+1:]
                gen_codes_list.append(stream)
                
            min_len = min([len(x) for x in gen_codes_list])
            gen_codes = [x[:min_len] for x in gen_codes_list]
            gen_codes = torch.stack(gen_codes, dim=0).unsqueeze(0).unsqueeze(0)
            
            if (gen_codes >= 2048).any():
                gen_codes[gen_codes >= 2048] = 0
            
            with torch.no_grad():
                gen_waveform = model.audio_encoder.decode(gen_codes, audio_scales=[None])
                gen_waveform = gen_waveform.audio_values.cpu()
            
            del outputs, logits, next_token_logits, probs, next_tokens
            torch.cuda.empty_cache()
        
        input_len_samples = target_segment.shape[-1]
        
        inference_samples = int(beat_duration * BEATS_PER_INFERENCE * SAMPLE_RATE)
        
        if gen_waveform.shape[-1] < input_len_samples + inference_samples:
             pad = (input_len_samples + inference_samples) - gen_waveform.shape[-1]
             gen_waveform = torch.nn.functional.pad(gen_waveform, (0, pad))
             
        new_segment = gen_waveform[:, :, input_len_samples : input_len_samples + inference_samples]
        
        full_target_audio = torch.cat([full_target_audio, new_segment.squeeze(0).to(device)], dim=1)
        
        print(f"Generated beats {i+1}-{i+BEATS_PER_INFERENCE}/{GENERATE_BEATS}")

    print("Saving output...")
    min_len = min(full_target_audio.shape[1], bg_mix.shape[1])
    final_mix = full_target_audio[:, :min_len] + bg_mix[:, :min_len]
    
    max_val = torch.max(torch.abs(final_mix))
    if max_val > 1.0:
        final_mix = final_mix / max_val
    
    out_path = os.path.join(OUTPUT_DIR, f"jam_{track_name}_{target_role}.wav")
    torchaudio.save(out_path, final_mix.cpu(), SAMPLE_RATE)
    
    out_path_target = os.path.join(OUTPUT_DIR, f"jam_{track_name}_{target_role}_solo.wav")
    torchaudio.save(out_path_target, full_target_audio.cpu(), SAMPLE_RATE)
    
    print(f"Saved mix to {out_path}")
    print(f"Saved solo target to {out_path_target}")

if __name__ == "__main__":
    main()
