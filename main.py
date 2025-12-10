#!/usr/bin/env python3
"""
Main entrypoint for the Physical AI & Humanoid Robotics Interactive Textbook
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Physical AI & Humanoid Robotics Interactive Textbook")
    parser.add_argument("--mode",
                       choices=["training", "simulation", "textbook", "evaluation"],
                       default="textbook",
                       help="Operating mode for the application")
    parser.add_argument("--config",
                       type=str,
                       default="configs/default_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--model-path",
                       type=str,
                       default=None,
                       help="Path to pre-trained model (for evaluation mode)")

    args = parser.parse_args()

    if args.mode == "training":
        from training.train_humanoid_rl import train_humanoid_rl
        train_humanoid_rl()
    elif args.mode == "simulation":
        print("Starting Isaac Sim environment...")
        # Launch Isaac Sim with humanoid robot
        pass
    elif args.mode == "textbook":
        print("Starting interactive textbook...")
        # Launch the Docusaurus textbook
        os.system("npm run dev")
    elif args.mode == "evaluation":
        if not args.model_path:
            print("Error: Model path required for evaluation mode")
            sys.exit(1)
        from training.evaluate_policy import evaluate_policy
        evaluate_policy(args.model_path)

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
