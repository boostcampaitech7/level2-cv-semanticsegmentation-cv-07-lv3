import os
import argparse
from train import main as train_main
from inference import main as inference_main

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and inference segmentation model')
    
    # Common arguments
    parser.add_argument('-c', '--config', type=str, default='smp_unetplusplus_efficientb0.yaml',
                        help='name of config file in configs directory')
    parser.add_argument('-s', '--save', type=str, default=None,
                        help='name of the model file to save. If not specified, uses the config name')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='name of the output file. If not specified, uses the save name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training from')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold for binary prediction')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # If save name is not specified, use the config name
    if args.save is None:
        args.save = os.path.splitext(args.config)[0] + '.pt'
    
    # Set model path
    args.model_path = os.path.join('checkpoints', args.save)
    
    # If output name is not specified, use the same name as the model
    if args.output is None:
        args.output = os.path.splitext(args.save)[0] + '.csv'
    args.output_path = os.path.join('results', args.output)
    
    return args

def main():
    """Main function to run training and inference sequentially"""
    args = parse_args()
    
    # Training
    print("Starting training...")
    train_args = argparse.Namespace(
        config=args.config,
        resume=args.resume
    )
    train_main(train_args)
    
    # Inference
    print("\nStarting inference...")
    inference_args = argparse.Namespace(
        config=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        threshold=args.threshold
    )
    inference_main(inference_args)

if __name__ == '__main__':
    main()