import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    TimesformerForVideoClassification, 
    TimesformerConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.trainer_utils import EvalPrediction
import optuna
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import json
import time
from PIL import Image
import torch.nn.functional as F
import logging
from datetime import datetime
from scipy.special import softmax
import pickle

import decord
from decord import VideoReader, cpu, gpu

# Suppress excessive logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

decord.bridge.set_bridge('torch')
LOG_DIR = "./logs_base_pruning"

class LogProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Only log dicts with 'loss' key (filter out Trainer's internal logs)
            if 'loss' in logs:
                logging.getLogger(__name__).info(logs)

def setup_logging():
    """Simplified logging setup"""
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/timesformer_optuna_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename)
        ],
        force=True  # Override existing loggers
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

class VideoDataset:
    def __init__(self, data_path, num_frames=8, image_size=224):
        self.data_path = data_path
        self.num_frames = num_frames
        self.image_size = image_size
        self.videos = []
        self.labels = []
        self._load_data()
        
        # Always use CPU context for Decord
        self.ctx = cpu(0)
        
        # ImageNet normalization constants on CPU
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _load_data(self):
        for label, class_dir in enumerate(sorted(os.listdir(self.data_path))):
            class_path = os.path.join(self.data_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                    self.videos.append(os.path.join(class_path, video_file))
                    self.labels.append(label)

    def _preprocess_video(self, video_path):
        """
        CPU-based video preprocessing using Decord, then move result to GPU.
        """
        try:
            vr = VideoReader(video_path, ctx=self.ctx)
            total = len(vr)
            if total < self.num_frames:
                logger.warning(f"Video {video_path} has only {total} frames, padding")
                indices = list(range(total)) + [total-1] * (self.num_frames - total)
            else:
                indices = np.linspace(0, total-1, self.num_frames, dtype=int).tolist()
            
            # Decord returns uint8 tensor on CPU
            frames = vr.get_batch(indices)  # shape [F,H,W,C]
            video = frames.permute(0,3,1,2).float() / 255.0  # [F,C,H,W], float CPU
            
            # Resize if needed
            if (video.shape[-1], video.shape[-2]) != (self.image_size, self.image_size):
                video = F.interpolate(video, size=(self.image_size, self.image_size),
                                      mode='bilinear', align_corners=False)
            
            # Normalize on CPU
            video = (video - self.mean) / self.std  # CPU tensor
            
            # Finally move to GPU if available
            if torch.cuda.is_available():
                video = video.cuda(non_blocking=True)
            
            return video
        
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.zeros(self.num_frames, 3, self.image_size, self.image_size,
                               device=device)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        pixel_values = self._preprocess_video(video_path)
        return {'pixel_values': pixel_values, 'labels': torch.tensor(label, dtype=torch.long)}

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predicted_probs = predictions  # Raw logits/probabilities
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    # ROC AUC for multiclass (one-vs-rest)
    try:
        # Convert logits to probabilities using softmax
        predicted_probs_softmax = softmax(predicted_probs, axis=1)
        roc_auc = roc_auc_score(labels, predicted_probs_softmax, multi_class='ovr', average='weighted')
    except:
        print("!!!!Warning: ROC AUC calculation failed, possibly due to single-class presence in batch.")
        logger.warning("ROC AUC calculation failed, possibly due to single-class presence in batch.")
        roc_auc = 0.0  # Fallback if ROC AUC calculation fails
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
    }

def create_model(num_labels=10, num_frames=8, dropout_rate=0.1):
    """Create TimeSformer model with proper configuration"""
    config = TimesformerConfig(
        num_labels=num_labels,
        num_frames=num_frames,
        image_size=224,
        patch_size=16,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate,
    )

    model = TimesformerForVideoClassification(config)
    return model

# def create_model(num_labels=7, num_frames=8, dropout_rate=0.1):
#     """Load pretrained TimeSformer model and fine-tune for your classes"""
#     logger.info("Loading pretrained TimeSformer model...")
    
#     # Load pretrained model
#     model = TimesformerForVideoClassification.from_pretrained(
#         "facebook/timesformer-base-finetuned-k400",  # Pretrained on Kinetics-400
#         num_labels=num_labels,                       
#         num_frames=num_frames,                       # Keep your frame count
#         ignore_mismatched_sizes=True                 # Allow different num_labels
#     )
    
#     # Update dropout rates if needed
#     if hasattr(model.config, 'hidden_dropout_prob'):
#         model.config.hidden_dropout_prob = dropout_rate
#         logger.info(f"Set hidden_dropout_prob to {dropout_rate}")
#     if hasattr(model.config, 'attention_probs_dropout_prob'):
#         model.config.attention_probs_dropout_prob = dropout_rate
#         logger.info(f"Set attention_probs_dropout_prob to {dropout_rate}")
    
#     logger.info(f"Pretrained model loaded with {num_labels} classes")
#     return model


class TrainingTimeTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.training_time = 0.0

    def start_training(self):
        self.start_time = time.time()

    def end_training(self):
        self.end_time = time.time()
        if self.start_time:
            self.training_time = self.end_time - self.start_time

    def get_training_time_minutes(self):
        return self.training_time / 60.0


# class OptunaPruningAndLoggingCallback(TrainerCallback):
#     def __init__(self, trial):
#         self.trial = trial

#     def on_evaluate(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             # Log key metrics
#             logger.info(f"Step {state.global_step} | Epoch {state.epoch:.2f} | "
#                        f"F1: {logs.get('eval_f1', 0):.4f} | "
#                        f"Acc: {logs.get('eval_accuracy', 0):.4f}")
            
#             # Optuna pruning
#             current_f1 = logs.get("eval_f1", logs.get("f1"))
#             if current_f1 is not None:
#                 self.trial.report(current_f1, step=state.global_step)
#                 if self.trial.should_prune():
#                     self.trial.set_user_attr("is_pruned", True)
#                     raise optuna.TrialPruned()

#         return control

#     def on_train_end(self, args, state, control, **kwargs):
#         if "is_pruned" not in self.trial.user_attrs:
#             self.trial.set_user_attr("is_pruned", False)
#         return control

class OptunaPruningAndLoggingCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Handle both logging and pruning when metrics are available"""
        if logs is not None:
            # Check if this is an evaluation log (has eval_ metrics)
            eval_metrics = {k: v for k, v in logs.items() if k.startswith('eval_')}
            
            if eval_metrics:  # This is evaluation logging
                # LOG METRICS
                logger.info("=== EVALUATION METRICS ===")
                
                f1 = logs.get('eval_f1', 0.0)
                acc = logs.get('eval_accuracy', 0.0)
                loss = logs.get('eval_loss', 0.0)
                
                logger.info(f"F1: {f1:.4f} | Accuracy: {acc:.4f} | Loss: {loss:.4f}")
                
                # OPTUNA PRUNING LOGIC
                current_f1 = logs.get('eval_f1')
                if current_f1 is not None:
                    # Report to Optuna
                    self.trial.report(current_f1, step=state.global_step)
                    logger.info(f"Reported F1={current_f1:.4f} to Optuna at step {state.global_step}")
                    
                    # Check if should prune
                    if self.trial.should_prune():
                        self.trial.set_user_attr("is_pruned", True)
                        logger.info(f"Trial {self.trial.number} pruned at step {state.global_step}")
                        raise optuna.TrialPruned(f"Trial pruned at step {state.global_step}")
                
                logger.info("=" * 50)
            
            elif 'loss' in logs:  # Training logs
                logger.info(f"Training - Step {state.global_step} | "
                           f"Epoch {state.epoch:.2f} | Loss: {logs['loss']:.4f}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Mark trial as completed if not pruned"""
        if "is_pruned" not in self.trial.user_attrs:
            self.trial.set_user_attr("is_pruned", False)
            logger.info(f"Trial {self.trial.number} completed successfully")
        return control



def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 2.88e-5, 1.81e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    weight_decay = trial.suggest_float('weight_decay', 1.1e-4, 9.72e-4, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0993, 0.342)
    num_epochs = trial.suggest_int('num_epochs', 10, 15)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.0999, 0.141)
    num_frames = trial.suggest_categorical('num_frames', [8, 10])

    scheduler_name = trial.suggest_categorical('scheduler_name', ['cosine', 'linear'])

    # Data paths 
    train_path = r"C:\Users\Student\Desktop\CML_HPO\dataset\UCF_Sports\UCF_train\train"
    val_path = r"C:\Users\Student\Desktop\CML_HPO\dataset\UCF_Sports\UCF_val\test"

    train_dataset = VideoDataset(train_path, num_frames=num_frames, image_size=224)
    val_dataset = VideoDataset(val_path, num_frames=num_frames, image_size=224)

    logger.info(f"Total training samples: {len(train_dataset)}, Total validation samples: {len(val_dataset)}")

    # Create model with proper TimeSformer configuration
    model = create_model(num_labels=10, num_frames=num_frames, dropout_rate=dropout_rate)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/trial_{trial.number}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=scheduler_name,
        warmup_ratio=warmup_ratio,

        # Logging and evaluation
        logging_steps=50,
        eval_steps=200,
        eval_strategy="steps",
        logging_dir=LOG_DIR,     # where to write logs
        logging_strategy="steps",          # Ensure step-wise logging


        # No model saving
        save_strategy="no",
        load_best_model_at_end=False,

        # Optimize for F1 score
        metric_for_best_model="eval_f1",
        greater_is_better=True,

        # Performance optimizations
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=True,
        report_to=None,
    )

    # Initialize training time tracker
    time_tracker = TrainingTimeTracker()

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[
            OptunaPruningAndLoggingCallback(trial),
            LogProgressCallback()
        ]
    )

    logger.info(f"Using device: {trainer.args.device}")

    try:

        params = trial.params
        logger.info(f"=== TRIAL {trial.number} STARTED ===")
        logger.info(f"Trial {trial.number} hyperparameters: {params}")


        time_tracker.start_training()
        trainer.train()
        time_tracker.end_training()

        eval_results = trainer.evaluate()
        f1_score_val = eval_results.get('eval_f1', eval_results.get('f1', 0.0))

        # LOG FINAL TRIAL RESULTS
        training_time_minutes = time_tracker.get_training_time_minutes()

        # Log all final metrics
        logger.info("=== FINAL METRICS ===")
        logger.info(f"Final F1 Score: {f1_score_val:.4f}")
        logger.info(f"Final Accuracy: {eval_results.get('eval_accuracy', 0.0):.4f}")
        logger.info(f"Final Precision: {eval_results.get('eval_precision', 0.0):.4f}")
        logger.info(f"Final Recall: {eval_results.get('eval_recall', 0.0):.4f}")
        logger.info(f"Final ROC AUC: {eval_results.get('eval_roc_auc', 0.0):.4f}")
        logger.info(f"Final Loss: {eval_results.get('eval_loss', 0.0):.4f}")
        
        # Log confusion matrix if available
        if 'eval_confusion_matrix' in eval_results:
            cm = np.array(eval_results['eval_confusion_matrix'])
            logger.info("=== FINAL CONFUSION MATRIX ===")
            for i, row in enumerate(cm):
                logger.info(f"Class {i}: {row}")

        # Store metrics
        training_time_minutes = time_tracker.get_training_time_minutes()
        trial.set_user_attr('training_time_minutes', training_time_minutes)
        trial.set_user_attr('final_accuracy', eval_results.get('eval_accuracy', 0.0))
        trial.set_user_attr('final_f1_score', f1_score_val)
        trial.set_user_attr('final_precision', eval_results.get('eval_precision', 0.0))
        trial.set_user_attr('final_recall', eval_results.get('eval_recall', 0.0))
        trial.set_user_attr('final_eval_loss', eval_results.get('eval_loss', 0.0))
        trial.set_user_attr('total_training_steps', trainer.state.global_step)
        
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()

        trial.set_user_attr('is_pruned', False)
        return f1_score_val

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        time_tracker.end_training()
        logger.info(f"Trial {trial.number} interrupted by user at step {trainer.state.global_step if 'trainer' in locals() else 0}")
        trial.set_user_attr('is_interrupted', True)
        trial.set_user_attr('training_time_minutes', time_tracker.get_training_time_minutes())
        trial.set_user_attr('interrupted_at_step', trainer.state.global_step if 'trainer' in locals() else 0)

        # Clean up
        try:
            del model
            del trainer
        except:
            pass
        torch.cuda.empty_cache()
        
        # Re-raise to propagate to study.optimize()
        raise

    except optuna.TrialPruned:
        time_tracker.end_training()
        trial.set_user_attr('is_pruned', True)
        trial.set_user_attr('training_time_minutes', time_tracker.get_training_time_minutes())
        trial.set_user_attr('pruned_at_step', trainer.state.global_step if 'trainer' in locals() else 0)

        del model
        del trainer
        torch.cuda.empty_cache()
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        time_tracker.end_training()
        trial.set_user_attr('training_time_minutes', time_tracker.get_training_time_minutes())
        trial.set_user_attr('error_message', str(e))

        try:
            del model
            del trainer
        except:
            pass
        torch.cuda.empty_cache()
        return 0.0

def main():
    
    # storage_name = "postgresql://postgres:WeDontKnow2026@db.qbukuvngwcdebqmwtrtc.supabase.co:5432/postgres"
    storage_name = "sqlite:///testing6_optuna_ucf_sports_base_percentile_pruning.db"    
    sampler_file = "ucf_sports_base_percentile.pkl"  # File to save sampler state

    # pruner = optuna.pruners.MedianPruner(
    #     n_startup_trials=8,     # Wait for 5 trials before pruning # 5
    #     n_warmup_steps=400,       # 400 steps
    #     interval_steps=1        # Check at every evaluation
    # )

    pruner = optuna.pruners.PercentilePruner(
        25.0,                   
        n_startup_trials=8,     
        n_warmup_steps=600,       
        interval_steps=1
    )

    # study = optuna.create_study(
    #     study_name="timeSformer_sportsClassification_7classKinetics_base_percentile_pruning",
    #     direction="maximize",
    #     sampler=optuna.samplers.TPESampler(seed=42),
    #     pruner=pruner,
    #     storage=storage_name,
    #     load_if_exists=True
    # )
    # Load existing sampler or create new one
    try:
        with open(sampler_file, "rb") as f:
            sampler = pickle.load(f)
        logger.info(f"Loaded sampler state from {sampler_file}")
        print(f"✓ Loaded sampler state from {sampler_file}")
    except FileNotFoundError:
        sampler = optuna.samplers.TPESampler(seed=42)
        logger.info("Created new sampler (no previous state found)")
        print("✓ Created new sampler")

    study = optuna.create_study(
        study_name="testingMAML6_timeSformer_sportsClassification_ucf_sports_base_percentile_pruning",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_name,
        load_if_exists=True
    )
   
    logger.info(f"Study created - Name: {study.study_name}")
    
    # Check progress
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Completed trials: {completed_trials}")
    print()
    
    n_trials = 20
    remaining_trials = n_trials - completed_trials

    print(f"Study: {study.study_name}")
    print(f"Remaining trials: {remaining_trials}")
    print()

    try:
        # CHANGED: Use ask-and-tell interface for fine-grained control
        for i in range(remaining_trials):
            # CRITICAL: Save sampler state BEFORE asking for next trial
            with open(sampler_file, "wb") as f:
                pickle.dump(study.sampler, f)
            logger.info(f"Sampler state saved BEFORE trial {completed_trials + i}")
            
            # Ask for a new trial
            trial = study.ask()
            logger.info(f"Starting trial {trial.number}")
            
            try:
                # Run the objective function
                value = objective(trial)
                
                # Tell the study the result
                study.tell(trial, value)
                logger.info(f"Trial {trial.number} completed with value {value:.4f}")
                
            except optuna.TrialPruned:
                # Handle pruned trials
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                logger.info(f"Trial {trial.number} was pruned")
                
            except KeyboardInterrupt:
                # User interrupted - mark trial as FAILED
                logger.info(f"Trial {trial.number} interrupted by user")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                print("\n✓ Study interrupted by user. Sampler state saved BEFORE this trial.")
                print("  When you resume, this trial will be re-run with the same parameters.")
                raise  # Re-raise to exit the loop
                
            except Exception as e:
                # Other errors - mark as FAILED
                logger.error(f"Trial {trial.number} failed with error: {e}")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    
    print("\nOptimization completed or interrupted!")
    
    # Check if there are any completed trials before accessing best_value
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_trials > 0:
        print(f"Best F1 score: {study.best_value:.4f}")
        print(f"Best trial: {study.best_trial.number}")

        # Log best hyperparameters
        logger.info("=== BEST HYPERPARAMETERS ===")
        for param, value in study.best_trial.params.items():
            logger.info(f"{param}: {value}")
        
        # Log best trial metrics
        logger.info("=== BEST TRIAL METRICS ===")
        for attr, value in study.best_trial.user_attrs.items():
            if isinstance(value, float):
                logger.info(f"{attr}: {value:.4f}")
            else:
                logger.info(f"{attr}: {value}")
    else:
        print("No completed trials yet.")

    print("Completed")

if __name__ == "__main__":
    main()