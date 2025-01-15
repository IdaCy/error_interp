import json
import argparse
import os

def remove_key_from_json(input_file, output_file, key_to_remove):
    # Verify if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist!")
        return

    # Load JSON data
    try:
        with open(input_file, 'r') as infile:
            data = json.load(infile)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Debug: Print the structure and number of entries
    print("Data type:", type(data))
    if isinstance(data, list):
        print("Number of entries:", len(data))
    else:
        print("JSON structure is not a list, it's:", type(data))
        return

    # Remove the specified key from each dictionary in the list
    for item in data:
        if isinstance(item, dict) and key_to_remove in item:
            print(f"Removing key '{key_to_remove}' from item with element_index {item.get('element_index')}")
            del item[key_to_remove]

    # Write the modified JSON data to the output file
    try:
        with open(output_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)
        print(f"Key '{key_to_remove}' has been removed from all entries in '{input_file}' and saved to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to output file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Remove a specified key from every dictionary in a JSON file."
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output JSON file")
    parser.add_argument("key", help="The key to remove from each dictionary")
    args = parser.parse_args()

    remove_key_from_json(args.input_file, args.output_file, args.key)

if __name__ == '__main__':
    main()
