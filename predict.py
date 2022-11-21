import os
import torch
from torchvision import models
from PIL import Image
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path',type=str, help='Path of directory of an image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file of the trained model')
    parser.add_argument('--gpu',action='store_true',help='Use GPU if available')
    parser.add_argument('--top_k', type = int, default=5,
                    help = 'K mostly class of the image prediction') 
    parser.add_argument('--category_names', type = str,
                    help = 'Category names json file')
    
    device = "cpu"

    args = parser.parse_args()
    
    if args.image_path:
        image_path = args.image_path
    if args.checkpoint:
        checkpoint = args.checkpoint
    if args.gpu:        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.top_k:
        top_k = args.top_k
    if args.category_names:
        category_names = args.category_names
        
    if not os.path.exists(image_path):
        raise Exception('Oops! Your image file does not exist!');
        
    if not os.path.exists(checkpoint):
        raise Exception('Oops! Your checkpiont file does not exist!');
    
    if category_names is not None and not os.path.exists(category_names):
        raise Exception('Oops! Your category name file does not exist!');
        
    model = load_checkpoint(checkpoint)
    model.to(device)
    
    prob, classes = predict(image_path, model, top_k, device)
    
    cat_to_name = None
    if category_names is not None:
        try:
            with open(category_names, 'r') as f:
                cat_to_name = json.load(f)
        except:
            raise Exception('Category name file is not valid')
            
    i = 0

    for prob, label in zip(prob, classes):
        i = i + 1
        if (cat_to_name):
            label = cat_to_name[label]
        else:
            label = 'Class {}'.format(str(label))
        print(f"{i}. {label} [{(prob*100):.3f}%]")
        
def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    else:
        model = models.densenet121(pretrained=True)
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer_dict']
    
    for param in model.parameters():
        param.requires_grad = False
    return model        

def process_image(image):
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Resize
    im_resize = image.resize((256,256));
    
    # Get dimensions
    width, height = im_resize.size   

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    im_crop = im_resize.crop((left, top, right, bottom))
    
    np_image = np.array(im_crop)
    np_image = np_image/255
    
    normalized_image = ( np_image - mean ) / std 

    transposed_image = normalized_image.transpose((2, 0, 1))

    return transposed_image

def predict(image_path, model, topk, device):
    
    image = Image.open(image_path)
    
    np_image = process_image(image)
    
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze(dim=0)
    
    tensor_image = tensor_image.to(device)
    
    model.eval()
    with torch.no_grad():
        
        output = model.forward(tensor_image)
        
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

        probs = list(top_p.cpu().numpy().squeeze())
           
        inverse_index = {model.class_to_idx[i]: i for i in model.class_to_idx}
        fclasses = list()
    
        for c in top_class.tolist()[0]:
            fclasses.append(inverse_index[c])
            
        print(fclasses)
            
    return probs, fclasses
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
