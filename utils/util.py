import cv2
import numpy as np
import paddle

def cv2paddle(img):
    '''
    cv2paddle:
    
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - converts the image from BGR to RGB
        np.transpose(img, (2, 0, 1)) - transposes the image from (H, W, C) to (C, H, W)
            paddle.to_tensor(img, dtype='float32').unsqueeze(axis=0) - converts the image from numpy array to tensor - unsqueezes the image to add a batch dimension
        img = img / 255.0 - normalizes the image by dividing by 255.0
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = paddle.to_tensor(img, dtype='float32').unsqueeze(axis=0)
    img = img / 255.0
    return img

def paddle2cv(img):
    '''
    paddle2cv:
    
        img = img.numpy()[0] - converts the image from tensor to numpy array and removes the batch dimension
        np.transpose(img, (1, 2, 0)) - transposes the image from (C, H, W) to (H, W, C)
        img *= 255 - multiplies the image by 255 to get back the original values of pixels in range [0-255]
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR) - converts the image from RGB to BGR for OpenCV compatibility
    '''
    
    img = img.numpy()[0]
    img = np.transpose(img, (1, 2, 0))
    img *= 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
