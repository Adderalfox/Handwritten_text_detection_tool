import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
from models.model import HandwrittenCNN
from utils.transforms import load_emnist_mapping
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HandwrittenCNN(num_classes=62)
model.load_state_dict(torch.load('checkpoint80.pth', map_location=device)['model_state_dict'])
model.to(device)
model.eval()


class ImprovedTextSegmenter:
    def __init__(self):
        pass

    def preprocess_image(self, image_path):
        """Preprocess the image for better segmentation with noise removal"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply morphological opening to remove small noise
        kernel = np.ones((2, 2), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 4)

        # Remove small noise using morphological operations
        # Remove small white noise
        kernel_noise = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_noise)

        # Remove small black noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)

        # Invert if text is dark on light background
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        return binary, gray

    def segment_lines_improved(self, binary_img):
        """Improved line segmentation using contours and clustering"""
        # Find all contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Get bounding boxes of all contours
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small contours (noise)
            if w > 5 and h > 5:
                bboxes.append((x, y, w, h, y + h // 2))  # Include center y for clustering

        if not bboxes:
            return []

        # Cluster contours by their vertical position (y-coordinate)
        centers_y = np.array([[bbox[4]] for bbox in bboxes])  # Use center y

        # Use DBSCAN clustering to group contours into lines
        clustering = DBSCAN(eps=20, min_samples=1).fit(centers_y)
        labels = clustering.labels_

        # Group contours by line
        lines_dict = {}
        for i, label in enumerate(labels):
            if label not in lines_dict:
                lines_dict[label] = []
            lines_dict[label].append(bboxes[i])

        # Sort lines by average y position
        lines = []
        for label, line_bboxes in lines_dict.items():
            avg_y = np.mean([bbox[1] for bbox in line_bboxes])

            # Get line boundaries
            min_y = min([bbox[1] for bbox in line_bboxes])
            max_y = max([bbox[1] + bbox[3] for bbox in line_bboxes])

            # Add padding
            padding = 5
            min_y = max(0, min_y - padding)
            max_y = min(binary_img.shape[0], max_y + padding)

            line_img = binary_img[min_y:max_y, :]
            lines.append((line_img, min_y, max_y, avg_y, line_bboxes))

        # Sort by average y position
        lines.sort(key=lambda x: x[3])

        return lines

    def segment_characters_contour_based(self, line_img, line_bboxes, line_start_y):
        """Segment characters using contour analysis"""
        # Find contours in the line
        contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Get character bounding boxes
        char_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on size (remove noise and very large objects)
            if 8 <= w <= 100 and 10 <= h <= 80:
                char_boxes.append((x, y, w, h))

        # Sort characters by x position (left to right)
        char_boxes.sort(key=lambda x: x[0])

        # Extract character images with better padding
        characters = []
        for x, y, w, h in char_boxes:
            # Add padding around character
            padding = 3
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(line_img.shape[1], x + w + padding)
            y_end = min(line_img.shape[0], y + h + padding)

            char_img = line_img[y_start:y_end, x_start:x_end]

            # Skip very small regions
            if char_img.shape[0] > 5 and char_img.shape[1] > 5:
                characters.append((char_img, x_start, y_start))

        return characters

    def merge_close_characters(self, characters, merge_threshold=5):
        """Merge characters that are very close together (likely part of same letter)"""
        if len(characters) <= 1:
            return characters

        merged_chars = []
        i = 0

        while i < len(characters):
            current_char = characters[i]
            current_img, current_x, current_y = current_char

            # Check if next character is very close
            if i + 1 < len(characters):
                next_char = characters[i + 1]
                next_img, next_x, next_y = next_char

                # Calculate distance between characters
                current_right = current_x + current_img.shape[1]
                distance = next_x - current_right

                # If characters are very close, merge them
                if distance < merge_threshold:
                    # Create merged bounding box
                    min_x = min(current_x, next_x)
                    max_x = max(current_x + current_img.shape[1], next_x + next_img.shape[1])
                    min_y = min(current_y, next_y)
                    max_y = max(current_y + current_img.shape[0], next_y + next_img.shape[0])

                    # Get the line image to extract merged character
                    # This assumes we have access to the full line image
                    # For now, we'll skip merging and take the larger character
                    if current_img.shape[0] * current_img.shape[1] > next_img.shape[0] * next_img.shape[1]:
                        merged_chars.append(current_char)
                    else:
                        merged_chars.append(next_char)

                    i += 2  # Skip both characters
                else:
                    merged_chars.append(current_char)
                    i += 1
            else:
                merged_chars.append(current_char)
                i += 1

        return merged_chars


class HandwrittenTextRecognizer:
    def __init__(self):
        self.segmenter = ImprovedTextSegmenter()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def center_image(self, img_array):
        """Center and preprocess character image for model input with EMNIST transformations"""
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.convert('L')

        # Apply the same transformations as you did for single character testing
        # 1. Flip left to right
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Rotate 90 degrees
        img = img.rotate(90)

        # 3. Invert colors (make text white on black background like EMNIST)
        # img = ImageOps.invert(img)

        # Find bounding box of non-zero pixels
        img_array = np.array(img)
        coords = np.column_stack(np.where(img_array > 50))

        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Add small padding
            padding = 2
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(img_array.shape[0], y_max + padding)
            x_max = min(img_array.shape[1], x_max + padding)

            # Crop to content
            cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))

            # Create square image with proper centering (like EMNIST format)
            max_side = max(cropped.size[0], cropped.size[1])
            # Add some padding to the square
            max_side = int(max_side * 1.2)

            new_img = Image.new('L', (max_side, max_side), color=0)  # Black background
            offset = ((max_side - cropped.size[0]) // 2, (max_side - cropped.size[1]) // 2)
            new_img.paste(cropped, offset)

            # Enhance contrast like you did before
            new_img = ImageEnhance.Contrast(new_img).enhance(2.0)

            return new_img

        return img

    def predict_character(self, char_img):
        """Predict a single character from image array with EMNIST preprocessing"""
        try:
            # Check if character image is too small or empty
            if char_img.shape[0] < 5 or char_img.shape[1] < 5:
                return '?'

            # Check if image is mostly empty
            if np.sum(char_img > 50) < 10:
                return '?'

            # Preprocess the character image with EMNIST transformations
            pil_img = self.center_image(char_img)

            # Apply transforms (same as your original single character prediction)
            tensor_img = self.transform(pil_img).unsqueeze(0).to(device)

            # Save transformed image for debugging (like in your original code)
            if hasattr(self, 'save_debug_transforms') and self.save_debug_transforms:
                unnormalized = tensor_img.squeeze(0) * 0.5 + 0.5
                to_pil = transforms.ToPILImage()
                img_out = to_pil(unnormalized.cpu())
                img_out.save(f"debug_transformed_{np.random.randint(1000)}.png")

            # Make prediction
            with torch.no_grad():
                outputs = model(tensor_img)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                # Only return prediction if confidence is above threshold
                if confidence.item() < 0.2:  # Lower threshold since we have proper preprocessing now
                    return '?'

                predicted_class = predicted.item()

                # Get character mapping
                mapping = load_emnist_mapping()
                predicted_char = mapping[predicted_class]

                return predicted_char
        except Exception as e:
            print(f"Error predicting character: {e}")
            return '?'

    def recognize_text(self, image_path, save_debug=False, save_debug_transforms=False):
        """Main function to recognize text from image"""
        print(f"Processing image: {image_path}")

        # Set flag for saving debug transforms
        self.save_debug_transforms = save_debug_transforms

        # Preprocess image
        binary_img, gray_img = self.segmenter.preprocess_image(image_path)

        if save_debug:
            cv2.imwrite('debug_binary.png', binary_img)
            cv2.imwrite('debug_gray.png', gray_img)
            print("Saved debug images")

        # Segment into lines using improved method
        lines = self.segmenter.segment_lines_improved(binary_img)
        print(f"Found {len(lines)} lines")

        recognized_text = []

        for line_idx, (line_img, line_start, line_end, avg_y, line_bboxes) in enumerate(lines):
            print(f"\nProcessing line {line_idx + 1}")

            if save_debug:
                cv2.imwrite(f'debug_line_{line_idx}.png', line_img)

            # Segment line into characters using contour-based method
            characters = self.segmenter.segment_characters_contour_based(
                line_img, line_bboxes, line_start
            )

            # Merge very close characters
            characters = self.segmenter.merge_close_characters(characters)

            print(f"Found {len(characters)} characters in line {line_idx + 1}")

            line_text = ""
            word_chars = []

            for char_idx, (char_img, char_x, char_y) in enumerate(characters):
                # Predict character with EMNIST preprocessing
                predicted_char = self.predict_character(char_img)

                if save_debug:
                    debug_path = f'debug_line{line_idx}_char{char_idx}_{predicted_char}.png'
                    cv2.imwrite(debug_path, char_img)

                # Check for word boundaries (large gaps between characters)
                if char_idx > 0:
                    prev_char = characters[char_idx - 1]
                    prev_char_img, prev_char_x, prev_char_y = prev_char
                    gap = char_x - (prev_char_x + prev_char_img.shape[1])

                    # If gap is large, it's likely a space between words
                    if gap > 20:  # Adjust threshold as needed
                        if word_chars:
                            line_text += ''.join(word_chars) + ' '
                            word_chars = []

                if predicted_char != '?':
                    word_chars.append(predicted_char)

            # Add remaining characters
            if word_chars:
                line_text += ''.join(word_chars)

            if line_text.strip():  # Only add non-empty lines
                recognized_text.append(line_text.strip())

        return '\n'.join(recognized_text)

    def visualize_segmentation(self, image_path):
        """Visualize the segmentation process"""
        binary_img, gray_img = self.segmenter.preprocess_image(image_path)
        lines = self.segmenter.segment_lines_improved(binary_img)

        # Create visualization
        fig, axes = plt.subplots(len(lines) + 2, 1, figsize=(15, 3 * (len(lines) + 2)))
        if len(lines) == 0:
            return

        # Show original and binary images
        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title('Original Grayscale Image')
        axes[0].axis('off')

        axes[1].imshow(binary_img, cmap='gray')
        axes[1].set_title('Binary Image')
        axes[1].axis('off')

        # Show segmented lines with character boxes
        for i, (line_img, line_start, line_end, avg_y, line_bboxes) in enumerate(lines):
            if i + 2 < len(axes):
                # Get characters for this line
                characters = self.segmenter.segment_characters_contour_based(
                    line_img, line_bboxes, line_start
                )

                # Draw character bounding boxes on line image
                line_img_color = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
                for char_img, char_x, char_y in characters:
                    cv2.rectangle(line_img_color,
                                  (char_x, char_y),
                                  (char_x + char_img.shape[1], char_y + char_img.shape[0]),
                                  (255, 0, 0), 1)

                axes[i + 2].imshow(line_img_color)
                axes[i + 2].set_title(f'Line {i + 1} with Character Boxes')
                axes[i + 2].axis('off')

        plt.tight_layout()
        plt.savefig('improved_segmentation_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved segmentation visualization as improved_segmentation_visualization.png")


# Main usage function
def recognize_handwritten_text(image_path, save_debug=False, visualize=False, save_debug_transforms=False):
    """
    Main function to recognize handwritten text from an image

    Args:
        image_path (str): Path to the image file
        save_debug (bool): Whether to save debug images
        visualize (bool): Whether to show segmentation visualization
        save_debug_transforms (bool): Whether to save transformed character images

    Returns:
        str: Recognized text
    """
    recognizer = HandwrittenTextRecognizer()

    if visualize:
        recognizer.visualize_segmentation(image_path)

    # image = cv2.imread(image_path)
    # debug_path = f'imagebeforeprediction.png'
    # cv2.imwrite(debug_path, image)

    recognized_text = recognizer.recognize_text(image_path, save_debug=save_debug,
                                                save_debug_transforms=save_debug_transforms)

    return recognized_text


if __name__ == '__main__':
    # Example usage
    test_image = 'sample4.jpg'  # Your form image

    print("Recognizing handwritten text with EMNIST preprocessing...")
    result = recognize_handwritten_text(test_image, save_debug=True, visualize=True,
                                        save_debug_transforms=True)

    print("\n" + "=" * 50)
    print("RECOGNIZED TEXT:")
    print("=" * 50)
    print(result)
    print("=" * 50)

    # Save result to file
    with open('recognized_text.txt', 'w') as f:
        f.write(result)
    print("\nSaved recognized text to 'recognized_text.txt'")