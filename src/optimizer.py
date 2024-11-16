import torch.optim as optim
from lion_pytorch import Lion

class OptimizerFactory:
    """Factory class for creating optimizers"""
    @staticmethod
    def get_optimizer(optimizer_config, model_parameters):
        optimizer_name = optimizer_config['NAME'].lower()
        lr = float(optimizer_config['LR'])
        weight_decay = float(optimizer_config['WEIGHT_DECAY'])
        
        if optimizer_name == 'adam':
            return optim.Adam(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('BETAS', (0.9, 0.999))
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('BETAS', (0.9, 0.999))
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                model_parameters,
                lr=lr,
                momentum=optimizer_config.get('MOMENTUM', 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_name == 'lion':
            return Lion(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('BETAS', (0.9, 0.99)),
                use_triton=optimizer_config.get('USE_TRITON', False)
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")