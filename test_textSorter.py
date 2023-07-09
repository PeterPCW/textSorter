from versions.textSorter_DBSCAN import sort_text
from tests.generate_mock_dataset import generate_mock_dataset

def run_test(test_name, size, similarity_level, lower_range, upper_range):
    # Generate a mock dataset with the specified size and line similarity level
    dataset = generate_mock_dataset(size=size, similarity_level=similarity_level)

    # Execute the sort_text() function
    sorted_text, num_clusters = sort_text(dataset)

    # Check the number of clusters
    assert lower_range <= num_clusters <= upper_range, f"Number of clusters ({num_clusters}) out of range ({lower_range}-{upper_range}) for {test_name}."

    # Check that all lines in the dataset are present in the sorted output
    dataset_lines = dataset.split('\n')
    sorted_lines = sorted_text.split('\n')
    assert all(line in sorted_lines for line in dataset_lines), f"All lines not present in the sorted output for {test_name}"

    print(f"{test_name} passed successfully with {num_clusters} clusters ({lower_range}-{upper_range}).")

# Run the test
run_test(test_name="Small Diverse Dataset", size=25, similarity_level=0.1, lower_range=5, upper_range=20)
run_test(test_name="Small Similar Dataset", size=25, similarity_level=0.333, lower_range=1, upper_range=10)
run_test(test_name="Large Diverse Dataset", size=150, similarity_level=0.05, lower_range=75, upper_range=140)
run_test(test_name="Large Medium Dataset", size=150, similarity_level=0.125, lower_range=7, upper_range=75)
run_test(test_name="Large Similar Dataset", size=150, similarity_level=0.333, lower_range=2, upper_range=25)