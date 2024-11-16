import torch.optim as optim

class OptimizerFactory:
    """Factory class for creating optimizers"""
    @staticmethod
    def get_optimizer(optimizer_config, model_parameters):
        optimizer_name = optimizer_config.NAME.lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                model_parameters,
                lr=optimizer_config['LR'],
                weight_decay=optimizer_config['WEIGHT_DECAY'],
                betas=optimizer_config.get('BETAS', (0.9, 0.999))
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                model_parameters,
                lr=optimizer_config['LR'],
                weight_decay=optimizer_config['WEIGHT_DECAY'],
                betas=optimizer_config.get('BETAS', (0.9, 0.999))
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                model_parameters,
                lr=optimizer_config['LR'],
                momentum=optimizer_config.get('MOMENTUM', 0.9),
                weight_decay=optimizer_config['WEIGHT_DECAY']
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")