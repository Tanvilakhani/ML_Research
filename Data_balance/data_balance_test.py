import os
import matplotlib.pyplot as plt
import seaborn as sns

def check_data_balance(data_dir):
    # Get class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Count images in each class
    class_counts = {}
    total_images = 0
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(image_files)
        class_counts[class_name] = count
        total_images += count
    
    # Print statistics
    print("\nDataset Balance Analysis:")
    print("-" * 50)
    print(f"Total number of images: {total_images}")
    print(f"Number of classes: {len(class_dirs)}")
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count} images ({percentage:.2f}%)")
    
    # Calculate imbalance
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    # Visualize distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('data_distribution_test.png')
    plt.show()

if __name__ == "__main__":
    DATA_DIR = r'C:\Users\admin\Documents\ML_research\MY_data\test'
    check_data_balance(DATA_DIR)