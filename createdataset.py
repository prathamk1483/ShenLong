import os
import cv2
import numpy as np
from collections import Counter
from random import sample

sources = "D:/Bloody Roar 2 project/Dataset/images"
source_files = os.listdir(sources)

one =  [1, 0, 0, 0, 0, 0]
two =  [0, 1, 0, 0, 0, 0]
three =[0, 0, 1, 0, 0, 0]
four = [0, 0, 0, 1, 0, 0]
five = [0, 0, 0, 0, 1, 0]
six =  [0, 0, 0, 0, 0, 1]

 
images = []
mappings = []
data = []

for file_name in source_files:
    image_path = os.path.join(sources, file_name)
    image = cv2.imread(image_path)

    cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.imshow(file_name, image)
    print(f"Showing image: {file_name}")
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
    # Wait for the user to input the choice
    choice = input("Enter the option for this situation (1, 2, 3, 4): ")

    images.append(image)

    if choice == '1':
        print(one)
        mappings.append(one)
    elif choice == '2':
        print(two)
        mappings.append(two)
    elif choice == '3':
        print(three)
        mappings.append(three)
    elif choice == '4':
        print(three)
        mappings.append(four)
    elif choice == '5':
        print(three)
        mappings.append(five)
    elif choice == '6':
        print(three)
        mappings.append(six)
        
    cv2.destroyAllWindows()

print("Mapping completed.")

# Convert lists to numpy arrays
images = np.array(images)
mappings = np.array(mappings)


gameplay_data = images
key_mappings = mappings

# Convert key mappings to tuples
key_mappings_tuples = [tuple(mapping) for mapping in key_mappings]

# Count occurrences of each key mapping
mapping_counts = Counter(map(tuple, key_mappings))

# Determine the minimum occurrence count
min_count = min(mapping_counts.values())

# Create a balanced dataset
balanced_data = []
for mapping, count in mapping_counts.items():
    # Find indices of frames with this key mapping
    indices = [i for i, x in enumerate(key_mappings_tuples) if x == mapping]
    # Randomly select a subset of frames equal to min_count
    balanced_indices = sample(indices, min_count)
    # Add balanced subset to the balanced dataset
    balanced_data.extend(zip(gameplay_data[balanced_indices], key_mappings[balanced_indices]))

# Shuffle the balanced dataset
np.random.shuffle(balanced_data)

# Separate balanced dataset into gameplay data and key mappings
balanced_gameplay_data, balanced_key_mappings = zip(*balanced_data)

# Save balanced dataset
np.save("balanced_gameplay_data.npy", np.array(balanced_gameplay_data))
np.save("balanced_key_mappings.npy", np.array(balanced_key_mappings))

print("Balanced dataset created and saved.")

