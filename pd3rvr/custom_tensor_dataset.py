from torch.utils.data import Dataset

'''
tensors: A tuple of three tensors, namely the image, label, and the bounding box coordinates.
transforms: A torchvision.transforms instance which will be used to process the image.
'''


class CustomTensorDataset(Dataset):
#init constructor
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms
    def __getitem__(self, index):
        #make tensor as image, label and bbox -- more in future
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        box = self.tensors[2][index]
        #transpose image to start with channel (channelXhXw) dimension instead of (hXwXchannel)
        image = image.permute(2,0,1)

        #apply transform id needed
        if self.transforms:
            image=self.transforms(image)
        #return tensor elements as tuple
        return (imae, label, bbox)
    def __len__(self):
        #return number of images in Dataset
        return self.tensors[0].size(0)
