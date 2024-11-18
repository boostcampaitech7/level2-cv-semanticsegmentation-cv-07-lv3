import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts
)

class SchedulerFactory:
    """Factory class for creating learning rate schedulers"""
    @staticmethod
    def get_scheduler(scheduler_config, optimizer):
        """
        Create a learning rate scheduler
        
        Args:
            scheduler_config: Dictionary containing scheduler configuration
            optimizer: The optimizer to schedule
            
        Returns:
            A learning rate scheduler
        """
        scheduler_name = scheduler_config['NAME'].lower()
        
        if scheduler_name == 'steplr':
            return StepLR(
                optimizer,
                step_size=scheduler_config['STEP_SIZE'],
                gamma=scheduler_config.get('GAMMA', 0.1)
            )
            
        elif scheduler_name == 'multisteplr':
            return MultiStepLR(
                optimizer,
                milestones=scheduler_config['MILESTONES'],
                gamma=scheduler_config.get('GAMMA', 0.1)
            )
            
        elif scheduler_name == 'exponentiallr':
            return ExponentialLR(
                optimizer,
                gamma=scheduler_config.get('GAMMA', 0.95)
            )
            
        elif scheduler_name == 'cosineannealinglr':
            return CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['T_MAX'],
                eta_min=scheduler_config.get('ETA_MIN', 0)
            )
            
        elif scheduler_name == 'reducelronplateau':
            return ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('MODE', 'min'),
                factor=scheduler_config.get('FACTOR', 0.1),
                patience=scheduler_config.get('PATIENCE', 10),
                verbose=scheduler_config.get('VERBOSE', True)
            )
            
        elif scheduler_name == 'cosineannealingwarmrestarts':
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config.get('T_MULT', 1),
                eta_min=scheduler_config.get('ETA_MIN', 0)
            )
            
        else:
            raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")