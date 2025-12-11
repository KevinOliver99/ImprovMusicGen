import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
import random
from torch.utils.data import DataLoader
from transformers import AutoProcessor, MusicgenForConditionalGeneration, get_cosine_schedule_with_warmup
from accelerate import Accelerator
import torch.optim as optim
from dataset import SlakhJamDataset, collate_fn
import os
from tqdm import tqdm
import math
import warnings
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

class Config:
    MODEL_PATH = "models/musicgen-small"
    DATA_DIR = "DATA/STEMS/SLAKH"
    TRAIN_CSV = "slakh_train_reclassified.csv"
    VAL_CSV = "slakh_val_reclassified.csv"
    OUTPUT_DIR = None   # Automatically named if "None"
    BATCH_SIZE = 32 
    EPOCHS = 700
    LEARNING_RATE = 2.0e-7
    WEIGHT_DECAY = 0.1
    BETAS = (0.9, 0.95) 
    WARMUP_STEPS = 13
    GRAD_ACCUM_STEPS = 64
    CONTEXT_BEATS = 20
    TARGET_BEATS = 20
    NUM_WORKERS = 16
    TRAIN_ROLE = None
    CFG_DROPOUT = 0.1
    TOKEN_CORRUPTION_RATE = 0.20
    
    RESUME_FROM_CHECKPOINT = None
    RESUME_EPOCH = 0

def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right. This was used to encode different EnCodec codebooks for training MusicGen, so we adopt it here.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    if pad_token_id is None:
        pad_token_id = 0

    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def main():
    if Config.TRAIN_ROLE:
        Config.OUTPUT_DIR = f"output_jam_lora_{Config.TRAIN_ROLE}"
    else:
        Config.OUTPUT_DIR = "output_jam_lora_all"
    
    print(f"--- CONFIGURATION ---")
    print(f"Training Role: {Config.TRAIN_ROLE}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")
    print(f"---------------------")
    
    print(f"Loading model from {Config.MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(Config.MODEL_PATH)
    model = MusicgenForConditionalGeneration.from_pretrained(Config.MODEL_PATH)
    
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
    
    model.audio_encoder.eval()
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("Applying LoRA Adapters...")

    target_modules = [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2" 
    ]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=128, 
        lora_alpha=256,
        lora_dropout=0.2,
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    Config.TRAIN_CSV = "slakh_train.csv"

    Config.VAL_CSV = "slakh_val.csv"

    train_dataset = SlakhJamDataset(
        csv_path=Config.TRAIN_CSV,
        stems_dir=Config.DATA_DIR,
        processor=processor,
        beats_context=Config.CONTEXT_BEATS,
        beats_target=Config.TARGET_BEATS,
        split="train",
        target_role=Config.TRAIN_ROLE
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS
    )

    val_loaders = {}
    if Config.TRAIN_ROLE:
        roles = [Config.TRAIN_ROLE]
    else:
        roles = ["Drums", "Bass", "Lead", "Harmony"]
    
    for role in roles:
        val_dataset = SlakhJamDataset(
            csv_path=Config.VAL_CSV,
            stems_dir=Config.DATA_DIR,
            processor=processor,
            beats_context=Config.CONTEXT_BEATS,
            beats_target=Config.TARGET_BEATS,
            split="val",
            target_role=role
        )
        
        loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS
        )
        
        loader = accelerator.prepare(loader)
        val_loaders[role] = loader

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=Config.BETAS
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_loader) / Config.GRAD_ACCUM_STEPS)
    num_training_steps = num_update_steps_per_epoch * Config.EPOCHS
    
    print(f"Total Training Steps: {num_training_steps} (Updates per Epoch: {num_update_steps_per_epoch})")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=num_training_steps
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    unwrapped_model = accelerator.unwrap_model(model)
    base_model = unwrapped_model
    depth = 0
    while (hasattr(base_model, "base_model") or hasattr(base_model, "model")) and depth < 10:
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model
        elif hasattr(base_model, "model"):
            base_model = base_model.model
        depth += 1
            
    if not hasattr(base_model, "enc_to_dec_proj"):
        raise AttributeError(f"Could not find enc_to_dec_proj in {type(base_model)}")
        
    global_enc_to_dec_proj = base_model.enc_to_dec_proj
    
    base_model.enc_to_dec_proj = torch.nn.Identity()

    log_path = os.path.join(Config.OUTPUT_DIR, "training_log.csv")
    if accelerator.is_main_process:
        if not os.path.exists(Config.OUTPUT_DIR):
            os.makedirs(Config.OUTPUT_DIR)
        
        mode = "a" if Config.RESUME_FROM_CHECKPOINT else "w"
        if not Config.RESUME_FROM_CHECKPOINT:
             with open(log_path, mode) as f:
                f.write("epoch,train_loss,val_loss,val_loss_drums,val_loss_bass,val_loss_lead,val_loss_harmony,lr\n")

    if Config.RESUME_FROM_CHECKPOINT:
        print(f"Resuming accelerator state from {Config.RESUME_FROM_CHECKPOINT}")
        
        adapter_path = os.path.join(Config.RESUME_FROM_CHECKPOINT, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            print(f"Loading LoRA adapters from {adapter_path}...")
            from safetensors.torch import load_file
            adapter_state = load_file(adapter_path)
            
            model.load_state_dict(adapter_state, strict=False)
        else:
            print("WARNING: No adapter_model.safetensors found in checkpoint!")

        original_models = accelerator._models

        accelerator._models = []
        
        accelerator.load_state(Config.RESUME_FROM_CHECKPOINT, strict=False)
        
        accelerator._models = original_models
        
        print(f"Resuming from epoch {Config.RESUME_EPOCH}")

    print("Starting training...")
    model.train()
    
    start_epoch = Config.RESUME_EPOCH
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, Config.EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                inputs = processor.feature_extractor(
                    raw_audio=batch['audio_prompts'],
                    sampling_rate=32000,
                    padding=True,
                    return_tensors="pt"
                )
                
                role_prompts_map = {
                    "Lead": "Lead.",
                    "Drums": "Drums.",
                    "Bass": "Bass.",
                    "Harmony": "Accompaniment."
                }
                
                text_prompts = [role_prompts_map.get(role, "") for role in batch['roles']]
                
                if Config.CFG_DROPOUT > 0:
                    text_prompts = [
                        "" if random.random() < Config.CFG_DROPOUT else prompt 
                        for prompt in text_prompts
                    ]
                
                text_inputs = processor(
                    text=text_prompts,
                    padding=True,
                    return_tensors="pt"
                )
                inputs["input_ids"] = text_inputs["input_ids"]
                inputs["attention_mask"] = text_inputs["attention_mask"]
                
                target_inputs = processor.feature_extractor(
                    raw_audio=batch['audio_targets'],
                    sampling_rate=32000,
                    padding=True,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                target_values = target_inputs["input_values"].to(accelerator.device)
                target_padding_mask = target_inputs["padding_mask"].to(accelerator.device)
                
                with torch.no_grad():
                    if inputs["input_values"].dim() == 2:
                        inputs["input_values"] = inputs["input_values"].unsqueeze(1)
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    audio_encoder = unwrapped_model.audio_encoder
                    decoder = unwrapped_model.decoder

                    prompt_outputs = audio_encoder.encode(inputs["input_values"])
                    prompt_codes = prompt_outputs.audio_codes.squeeze(0) 
                    
                    if target_values.dim() == 2:
                        target_values = target_values.unsqueeze(1)

                    target_outputs = audio_encoder.encode(target_values)
                    target_codes = target_outputs.audio_codes.squeeze(0)
                
                embed_tokens = decoder.model.decoder.embed_tokens
                
                prompt_embeds = torch.zeros(
                    prompt_codes.shape[0], 
                    prompt_codes.shape[2], 
                    decoder.config.hidden_size, 
                    device=accelerator.device
                )
                for i in range(decoder.num_codebooks):
                    codebook_indices = prompt_codes[:, i, :]
                    codebook_embeds = embed_tokens[i](codebook_indices)
                    prompt_embeds += codebook_embeds
                
                text_outputs = model.text_encoder(
                    input_ids=inputs["input_ids"].to(accelerator.device),
                    attention_mask=inputs["attention_mask"].to(accelerator.device)
                )
                text_embeds = text_outputs.last_hidden_state
                
                text_embeds = global_enc_to_dec_proj(text_embeds)
                
                prompt_mask = torch.ones(prompt_embeds.shape[:2], device=accelerator.device)
                
                encoder_hidden_states = torch.cat([text_embeds, prompt_embeds], dim=1)
                encoder_attention_mask = torch.cat([inputs["attention_mask"].to(accelerator.device), prompt_mask], dim=1)
                
                labels = target_codes
                
                pad_id = decoder.config.pad_token_id
                num_codebooks = labels.shape[1]
                seq_len = labels.shape[2]
                max_len = seq_len + num_codebooks
                
                labels_flat = labels.reshape(-1, seq_len)
                delayed_labels, _ = decoder.build_delay_pattern_mask(
                    labels_flat, 
                    pad_token_id=pad_id,
                    max_length=max_len
                )
                delayed_labels = delayed_labels.reshape(labels.shape[0], num_codebooks, -1)
                
                decoder_start_token_id = decoder.config.decoder_start_token_id
                decoder_input_ids = delayed_labels.new_zeros(delayed_labels.shape)
                decoder_input_ids[:, :, 1:] = delayed_labels[:, :, :-1].clone()
                decoder_input_ids[:, :, 0] = decoder_start_token_id
                
                if Config.TOKEN_CORRUPTION_RATE > 0:
                    corruption_mask = torch.rand(decoder_input_ids.shape, device=accelerator.device) < Config.TOKEN_CORRUPTION_RATE
                    
                    random_tokens = torch.randint(0, 2048, decoder_input_ids.shape, device=accelerator.device)
                    
                    corrupted_input_ids = torch.where(corruption_mask, random_tokens, decoder_input_ids)
                    
                    corrupted_input_ids[decoder_input_ids == decoder_start_token_id] = decoder_start_token_id
                    corrupted_input_ids[decoder_input_ids == pad_id] = pad_id
                    
                    model_input_ids = corrupted_input_ids
                else:
                    model_input_ids = decoder_input_ids
                
                outputs = model(
                    decoder_input_ids=model_input_ids,
                    labels=delayed_labels.transpose(1, 2),
                    encoder_outputs=(encoder_hidden_states,),
                    attention_mask=encoder_attention_mask
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss}")
        
        base_model.enc_to_dec_proj = global_enc_to_dec_proj
        
        model.eval()
        print("Starting validation...")
        
        val_losses = {}
        
        for role, val_loader in val_loaders.items():
            total_role_loss = 0
            print(f"Validating Role: {role}")
            val_progress_bar = tqdm(val_loader, disable=not accelerator.is_local_main_process)
            
            for batch in val_progress_bar:
                with torch.no_grad():
                    inputs = processor.feature_extractor(
                        raw_audio=batch['audio_prompts'],
                        sampling_rate=32000,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    role_prompts_map = {
                        "Lead": "Lead.",
                        "Drums": "Drums.",
                        "Bass": "Bass.",
                        "Harmony": "Accompaniment."
                    }
                    text_prompts = [role_prompts_map.get(r, "") for r in batch['roles']]
                    
                    text_inputs = processor(
                        text=text_prompts,
                        padding=True,
                        return_tensors="pt"
                    )
                    inputs["input_ids"] = text_inputs["input_ids"]
                    inputs["attention_mask"] = text_inputs["attention_mask"]
                    
                    target_inputs = processor.feature_extractor(
                        raw_audio=batch['audio_targets'],
                        sampling_rate=32000,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                    target_values = target_inputs["input_values"].to(accelerator.device)
                    target_padding_mask = target_inputs["padding_mask"].to(accelerator.device)
                    
                    if inputs["input_values"].dim() == 2:
                        inputs["input_values"] = inputs["input_values"].unsqueeze(1)
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    audio_encoder = unwrapped_model.audio_encoder

                    prompt_outputs = audio_encoder.encode(inputs["input_values"])
                    prompt_codes = prompt_outputs.audio_codes

                    if target_values.dim() == 2:
                        target_values = target_values.unsqueeze(1)
                    
                    target_outputs = audio_encoder.encode(target_values)
                    target_codes = target_outputs.audio_codes
                    
                    labels = torch.cat([prompt_codes, target_codes], dim=3)
                    
                    if labels.dim() == 4 and labels.shape[0] == 1:
                        labels = labels.squeeze(0)
                    if labels.dim() == 4:
                        if labels.shape[1] == 1:
                            labels = labels.squeeze(1)
                        elif labels.shape[3] == 1:
                            labels = labels.squeeze(3)
                    
                    labels_for_input = labels.clone()
                    labels_for_loss = labels.clone()
                    
                    prompt_len = prompt_codes.shape[-1]
                    labels_for_loss[:, :, :prompt_len] = -100

                    mask_float = target_padding_mask.float().unsqueeze(1)
                    target_len = target_codes.shape[-1]
                    mask_downsampled = torch.nn.functional.interpolate(
                        mask_float, size=target_len, mode='nearest'
                    )
                    mask_expanded = mask_downsampled.expand(-1, labels.shape[1], -1)
                    mask_expanded = mask_expanded.to(labels.device)

                    target_slice = labels_for_loss[:, :, prompt_len:]
                    if mask_expanded.shape[-1] != target_slice.shape[-1]:
                        mask_downsampled = torch.nn.functional.interpolate(
                            mask_float, size=target_slice.shape[-1], mode='nearest'
                        )
                        mask_expanded = mask_downsampled.expand(-1, labels.shape[1], -1)
                        mask_expanded = mask_expanded.to(labels.device)
                    
                    target_slice = target_slice.masked_fill(mask_expanded == 0, -100)
                    labels_for_loss[:, :, prompt_len:] = target_slice

                    unwrapped_model = accelerator.unwrap_model(model)
                    decoder = unwrapped_model.decoder
                    pad_id = unwrapped_model.config.decoder.pad_token_id
                    
                    num_codebooks = labels.shape[1]
                    seq_len = labels.shape[2]
                    max_len = seq_len + num_codebooks
                    
                    def apply_delay(lbls):
                        lbls_flat = lbls.reshape(-1, seq_len)
                        delayed, _ = decoder.build_delay_pattern_mask(
                            lbls_flat, 
                            pad_token_id=pad_id,
                            max_length=max_len
                        )
                        return delayed.reshape(lbls.shape[0], num_codebooks, -1)

                    delayed_input = apply_delay(labels_for_input)
                    delayed_loss = apply_delay(labels_for_loss)
                    
                    delayed_loss[delayed_loss == pad_id] = -100
                    
                    decoder_start_token_id = unwrapped_model.config.decoder.decoder_start_token_id
                    decoder_input_ids = delayed_input.new_zeros(delayed_input.shape)
                    decoder_input_ids[:, :, 1:] = delayed_input[:, :, :-1].clone()
                    decoder_input_ids[:, :, 0] = decoder_start_token_id
                    
                    model_input_ids = decoder_input_ids.reshape(-1, seq_len)

                    outputs = model(
                        input_ids=inputs["input_ids"].to(accelerator.device),
                        attention_mask=inputs["attention_mask"].to(accelerator.device),
                        decoder_input_ids=model_input_ids.to(accelerator.device),
                        labels=delayed_loss.transpose(1, 2)
                    )

                    unwrapped_model = accelerator.unwrap_model(model)
                    decoder = unwrapped_model.decoder
                    
                    num_codebooks = labels.shape[1]
                    seq_len = labels.shape[2]
                    max_len = seq_len + num_codebooks
                    
                    labels_flat = labels.reshape(-1, seq_len)
                    pad_id = unwrapped_model.config.decoder.pad_token_id
                    delayed_labels, _ = decoder.build_delay_pattern_mask(
                        labels_flat, 
                        pad_token_id=pad_id,
                        max_length=max_len
                    )
                    delayed_labels = delayed_labels.reshape(labels.shape[0], num_codebooks, -1)

                    outputs = model(
                        input_ids=inputs["input_ids"].to(accelerator.device),
                        attention_mask=inputs["attention_mask"].to(accelerator.device),
                        labels=delayed_labels.transpose(1, 2)
                    )
                    
                    loss = outputs.loss
                    total_role_loss += loss.item()
                    val_progress_bar.set_description(f"Val Loss ({role}): {loss.item():.4f}")
            
            avg_role_loss = total_role_loss / len(val_loader)
            val_losses[role] = avg_role_loss
            print(f"Epoch {epoch} Average Val Loss ({role}): {avg_role_loss}")

        avg_val_loss = sum(val_losses.values()) / len(val_losses)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                drum_loss = val_losses.get('Drums', '')
                bass_loss = val_losses.get('Bass', '')
                lead_loss = val_losses.get('Lead', '')
                harmony_loss = val_losses.get('Harmony', '')
                f.write(f"{epoch},{avg_loss},{avg_val_loss},{drum_loss},{bass_loss},{lead_loss},{harmony_loss},{current_lr}\n")

        model.train()
        
        base_model.enc_to_dec_proj = torch.nn.Identity()
        
        if accelerator.is_main_process:
            epoch_path = os.path.join(Config.OUTPUT_DIR, f"checkpoint_epoch_{epoch}")
            accelerator.save_state(epoch_path)
            model.save_pretrained(epoch_path)
            processor.save_pretrained(epoch_path)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(Config.OUTPUT_DIR, "checkpoint_best")
                accelerator.save_state(best_path)
                model.save_pretrained(best_path)
                processor.save_pretrained(best_path)
                print(f"New best model saved with val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
