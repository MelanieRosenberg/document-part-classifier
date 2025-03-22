from typing import List, Optional, Dict
from dataclasses import dataclass
from peft import LoraConfig, TaskType, PeftType

@dataclass
class LoRAParameters:
    """Parameters for LoRA configuration."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    bias: str = "none"
    target_modules: List[str] = None
    task_type: TaskType = TaskType.TOKEN_CLASSIFICATION

def get_default_target_modules(model_type: str) -> List[str]:
    """Get default target modules for different model types."""
    if "bert" in model_type.lower():
        return ["query", "key", "value"]
    elif "roberta" in model_type.lower():
        return ["q_proj", "k_proj", "v_proj"]
    elif "gpt" in model_type.lower():
        return ["c_attn"]
    else:
        return ["query", "key", "value"]  # Default to transformer attention modules

def create_lora_config(
    model_type: str,
    params: Optional[LoRAParameters] = None,
    inference_mode: bool = False
) -> LoraConfig:
    """Create LoRA configuration for a specific model type."""
    if params is None:
        params = LoRAParameters()
    
    if params.target_modules is None:
        params.target_modules = get_default_target_modules(model_type)
    
    return LoraConfig(
        task_type=params.task_type,
        inference_mode=inference_mode,
        r=params.r,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        bias=params.bias,
        target_modules=params.target_modules
    )

def get_optimal_lora_params(
    model_type: str,
    sequence_length: int,
    batch_size: int,
    available_memory_mb: float
) -> LoRAParameters:
    """Get optimal LoRA parameters based on available resources."""
    # Base memory usage estimation (very rough approximation)
    base_memory = sequence_length * batch_size * 4  # 4 bytes per float
    
    # Adjust rank based on available memory
    if available_memory_mb * 1024 * 1024 < base_memory * 2:
        # Limited memory - use smaller rank
        r = 8
        alpha = 16
        dropout = 0.2
    else:
        # More memory available - use larger rank
        r = 16
        alpha = 32
        dropout = 0.1
    
    return LoRAParameters(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=get_default_target_modules(model_type)
    )

def print_lora_info(config: LoraConfig) -> None:
    """Print information about LoRA configuration."""
    print("\nLoRA Configuration:")
    print(f"Rank (r): {config.r}")
    print(f"Alpha: {config.lora_alpha}")
    print(f"Dropout: {config.lora_dropout}")
    print(f"Target modules: {config.target_modules}")
    print(f"Task type: {config.task_type}")
    print(f"Inference mode: {config.inference_mode}")
    
def get_memory_efficient_config(model_type: str) -> LoraConfig:
    """Get memory-efficient LoRA configuration for resource-constrained environments."""
    return create_lora_config(
        model_type,
        LoRAParameters(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=get_default_target_modules(model_type)
        )
    ) 