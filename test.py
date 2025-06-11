import torch
from torchvision import transforms
from PIL import Image
from models.model import HandwrittenCNN
from utils.transforms import load_emnist_mapping
from PIL import ImageOps, ImageEnhance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HandwrittenCNN(num_classes=62)
# model.load_state_dict(torch.load('handwritten_cnn.pth', map_location=device))
model.load_state_dict(torch.load('checkpoint40.pth', map_location=device)['model_state_dict'])
model.to(device)
model.eval()

def center_image(img):
    img = img.convert('L')

    inverted = ImageOps.invert(img)
    thresholded = inverted.point(lambda x: 0 if x < 30 else 255, '1')

    bbox = thresholded.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        max_side = max(cropped.size)
        new_img = Image.new('L', (max_side, max_side), color=255)
        offset = ((max_side - cropped.size[0]) // 2, (max_side - cropped.size[1]) // 2)
        new_img.paste(cropped, offset)

        new_img = ImageEnhance.Contrast(new_img).enhance(3.0)

        return new_img.resize((28, 28), Image.LANCZOS)
    return img

class Invert:
    def __call__(self, tensor):
        return 1.0 - tensor

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    # Invert(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    img = Image.open(image_path)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(90)
    # img = np.flip(img, axis=1)

    img = center_image(img)
    transformed_tensor = transform(img) # image tranformation for visual representation after transform
    img = transform(img).unsqueeze(0).to(device)

    # this section saves the image after transform
    unnormalized = transformed_tensor * 0.5 + 0.5
    to_pil = transforms.ToPILImage()
    img_out = to_pil(unnormalized)
    img_out.save("transformed_visual.png")

    print('Saved transformed image as transformed_visual.png')

    # --------------------------------------------------------

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

        mapping = load_emnist_mapping()
        predicted_char = mapping[predicted_class]
        return predicted_char


if __name__ == '__main__':
    test_image = 'test6.png'
    prediction = predict_image(test_image)
    print(f'Predicted Character: {prediction}')