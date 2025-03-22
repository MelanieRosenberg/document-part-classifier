from typing import Dict, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm.auto import tqdm
import logging
from sklearn.metrics import f1_score, precision_recall_fscore_support
import psutil
import time
from ..models.model import DocumentClassifier

class DocumentTrainer:
    """Trainer for document part classification with mixed precision and monitoring."""
    
    def __init__(
        self,
        model: DocumentClassifier,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = True,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        model_save_path: Optional[str] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision and device == "cuda"
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.model_save_path = model_save_path
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Setup scheduler
        num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.global_step = 0
        self.best_val_f1 = 0.0
        self.training_history = {
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
            "learning_rates": [],
            "memory_usage": []
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model with mixed precision and monitoring."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_metrics = self._train_epoch()
            self.training_history["train_loss"].extend(train_metrics["losses"])
            self.training_history["train_f1"].append(train_metrics["f1"])
            
            # Validation
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                self.training_history["val_loss"].append(val_metrics["loss"])
                self.training_history["val_f1"].append(val_metrics["f1"])
                
                # Save best model
                if val_metrics["f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1"]
                    if self.model_save_path:
                        self._save_checkpoint(f"{self.model_save_path}_best.pt")
            
            # Save epoch checkpoint
            if self.model_save_path:
                self._save_checkpoint(f"{self.model_save_path}_epoch_{epoch+1}.pt")
        
        training_time = time.time() - start_time
        self.logger.info(f"\nTraining completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        losses = []
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc="Training",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                outputs = self.model(**batch)
                loss = outputs["loss"] / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if gradient accumulation is complete
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Record metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            losses.append(loss.item() * self.gradient_accumulation_steps)
            
            if "predictions" in outputs:
                all_preds.extend(outputs["predictions"].cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log memory usage
            if (step + 1) % self.logging_steps == 0:
                memory_used = psutil.Process().memory_info().rss / 1024**2
                self.training_history["memory_usage"].append(memory_used)
                self.logger.info(f"\nMemory usage: {memory_used:.1f}MB")
            
            # Evaluate if needed
            if self.val_dataloader is not None and (step + 1) % self.eval_steps == 0:
                val_metrics = self.evaluate()
                self.model.train()
                
                self.logger.info(
                    f"\nStep {self.global_step}: "
                    f"Val Loss = {val_metrics['loss']:.4f}, "
                    f"Val F1 = {val_metrics['f1']:.4f}"
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_dataloader)
        f1 = f1_score(all_labels, all_preds, average="weighted") if all_preds else 0.0
        
        return {
            "loss": avg_loss,
            "f1": f1,
            "losses": losses
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            total_loss += outputs["loss"].item()
            
            if "predictions" in outputs:
                all_preds.extend(outputs["predictions"].cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)
        
        metrics = {
            "loss": avg_loss,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
        
        if all_preds:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels,
                all_preds,
                average="weighted"
            )
            metrics.update({
                "f1": f1,
                "precision": precision,
                "recall": recall
            })
        
        return metrics
    
    def _save_checkpoint(self, path: str) -> None:
        """Save a model checkpoint."""
        torch.save({
            'epoch': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history
        }, path)
        self.logger.info(f"Saved checkpoint to: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['epoch']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.training_history = checkpoint['training_history']
        self.logger.info(f"Loaded checkpoint from: {path}") 