import json

def convert_from_labelme_format(labelme_path, original_path, output_path):
    """
    Convert LabelMe JSON data back to a general JSON format.
    
    Args:
        input_path (str): Path to the LabelMe JSON file.
        output_path (str): Path to save the converted JSON file.
    """
    with open(labelme_path, "r") as infile:
        labelme_data = json.load(infile)

    with open(original_path, "r") as originalfile:
        original_data = json.load(originalfile)
    
    original_annotations = {ann["label"]: ann for ann in original_data.get("annotations", [])}
    annotations = []
    
    for shape in labelme_data.get("shapes", []):
        label = shape.get("label", "unknown")
        points = shape.get("points", [])
        int_points = [[int(round(x)), int(round(y))] for x, y in points]
        
        # Find matching data in original_annotations
        original_annotation = original_annotations.get(label, {})
        
        # Create the merged annotation
        annotation = {
            "id": original_annotation.get("id", ""),  # Use ID from original or empty
            "type": original_annotation.get("type", "poly_seg"),  # Default type
            "attributes": original_annotation.get("attributes", {}),  # Default attributes
            "points": int_points,  # Points from LabelMe data
            "label": label  # Label from LabelMe data
        }
        annotations.append(annotation)
    
    # Create the final JSON structure
    merged_json = {
        "annotations": annotations
    }

    # Save the converted JSON
    with open(output_path, "w") as outfile:
        json.dump(merged_json, outfile, indent=4)
        print(f"Converted JSON saved to: {output_path}")


# Example usage
input_labelme_file = '../image1664241503868.json'
original_file = '/data/ephemeral/home/data/train/outputs_json/ID309/image1664241503868.json'
output_general_file = '/data/ephemeral/home/data/train/outputs_json/ID309/image1664241503868_1.json'
convert_from_labelme_format(input_labelme_file, original_file, output_general_file)
