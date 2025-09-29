#!/usr/bin/env python3
"""
ONNX Model Testing Script for Document Layout YOLO Model

This script tests whether the ONNX model loads correctly and can perform inference.
It creates dummy input data and runs a test inference to verify the model works.
"""

import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

def test_onnx_model(model_path):
    """
    Test ONNX model by loading it and running inference with dummy data.

    Args:
        model_path (str): Path to the ONNX model file
    """
    try:
        print(f"Loading ONNX model from: {model_path}")
        print("=" * 50)

        # Load the ONNX model
        session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully!")

        # Display model information
        print("\nModel Information:")
        print("-" * 30)

        # Input information
        inputs = session.get_inputs()
        print(f"Number of inputs: {len(inputs)}")
        for i, input_info in enumerate(inputs):
            print(f"Input {i}: {input_info.name}")
            print(f"  Shape: {input_info.shape}")
            print(f"  Type: {input_info.type}")

        # Output information
        outputs = session.get_outputs()
        print(f"\nNumber of outputs: {len(outputs)}")
        for i, output_info in enumerate(outputs):
            print(f"Output {i}: {output_info.name}")
            print(f"  Shape: {output_info.shape}")
            print(f"  Type: {output_info.type}")

        # Create dummy input data
        print("\nPreparing dummy input data...")
        print("-" * 30)

        # For YOLO models, typically expect input shape like [batch_size, channels, height, width]
        # Based on the model name, it seems to expect 1024x1024 input
        batch_size = 1
        channels = 3  # RGB
        height = 1024
        width = 1024

        # Create dummy input - normalized to [0,1] range as expected by most models
        dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        print(f"Created dummy input with shape: {dummy_input.shape}")
        print(f"Input data type: {dummy_input.dtype}")
        min_val = dummy_input.min()
        max_val = dummy_input.max()
        print(f'Input range: [{min_val:.4f}, {max_val:.4f}]')

        # Run inference
        print("\nRunning inference...")
        print("-" * 20)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        # Run inference
        results = session.run(output_names, {input_name: dummy_input})

        # Display results
        print("✅ Inference completed successfully!")
        print(f"\nNumber of outputs: {len(results)}")
        for i, result in enumerate(results):
            print(f"Output {i} shape: {result.shape}")
            print(f"Output {i} type: {result.dtype}")
            min_val = result.min()
            max_val = result.max()
            print(f'Output {i} range: [{min_val:.4f}, {max_val:.4f}]')

        print("\n" + "=" * 50)
        print("🎉 Model test completed successfully!")
        print("Your ONNX model is working properly.")

        return True

    except Exception as e:
        print(f"❌ Error during model testing: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the model test."""
    # Path to the ONNX model
    model_path = "/Users/likhithbhargav/Desktop/test onnx/doclayout_yolo_docstructbench_imgsz1024 (1).onnx"

    # Check if model file exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure the model file exists in the specified location.")
        sys.exit(1)

    # Run the test
    success = test_onnx_model(model_path)

    if success:
        print("\n✅ All tests passed! Your model is ready to use.")
    else:
        print("\n❌ Model test failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
