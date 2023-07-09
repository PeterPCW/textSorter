from versions.textSorter_DBSCAN import sort_text

def process_file(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Sort the text
    sorted_text, num_clusters = sort_text(file_content)

    # Ask the user whether to append or overwrite the file
    choice = input("Do you want to append the sorted lines to the file (A) or overwrite it (O)? ")
    if choice.lower() == "a":
        # Append the sorted text to the original file
        with open(file_path, 'a') as file:
            file.write('\n\n\n' + sorted_text)
    elif choice.lower() == "o":
        # Overwrite the file with the sorted text
        with open(file_path, 'w') as file:
            file.write(sorted_text)

    print(f"Sorting and writing to the file completed successfully with {num_clusters} clusters.")

# Ask the user for the file path
file_path = input("Enter the path to the .txt file: ")

# Process the file
process_file(file_path)
