import json
import os
import base64
import cv2

def convert_to_labelme_format(input_path, output_path, image_path, image_height, image_width):
    """
    Convert JSON data to LabelMe valid JSON format with points as float and include imageData.
    
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to save the LabelMe compatible JSON file.
        image_path (str): Path to the associated image.
        image_height (int): Height of the associated image.
        image_width (int): Width of the associated image.
    """
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        return

    with open(input_path, "r") as file:
        data = json.load(file)
    
    annotations = data.get("annotations", [])
    shapes = []

    for ann in annotations:
        label = ann.get("label", "unknown")
        points = ann.get("points", [])

        # Convert points to float format
        float_points = [[float(coord[0]), float(coord[1])] for coord in points]

        # Convert annotation to LabelMe's shape format
        shape = {
            "label": label,
            "points": float_points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    # Read the image and encode it as base64
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
    else:
        print(f"Image file not found: {image_path}")
        image_data = None  # Optional: Set to None if the image is missing

    # Create LabelMe compatible JSON structure
    labelme_data = {
        "version": "5.0.1",  # Adjust this to the actual LabelMe version if needed
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,  # Include base64 image data
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # Save the new JSON file
    with open(output_path, "w") as output_file:
        json.dump(labelme_data, output_file, indent=4)
    print(f"Converted JSON saved to {output_path}")


convert_to_labelme_format(
    input_path="/data/ephemeral/home/data/train/outputs_json/ID309/image1664241503868.json",
    output_path="../image1664241503868.json",
    image_path="/data/ephemeral/home/data/train/DCM/ID309/image1664241503868.png",
    image_height=2048,
    image_width=2048,
)

