import albumentations as alb
import keras
import numpy as np

class DataGenerator_albumentations(keras.utils.Sequence):
    def __init__(self, dir_x, dir_y, file_path_list, labels, batch_size, img_size, train):
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.file_path_list = file_path_list
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.train = train
        self.n = 0
        self.max = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.file_path_list)) / self.batch_size)

    def __getitem__(self, index):
        'generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # get list of IDs
        file_path_list_temp = [self.file_path_list[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(file_path_list_temp)
        # return data
        return X, y

    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.file_path_list))
        np.random.shuffle(self.indexes)

    def __data_generation(self, file_path_list_temp):
        'generate data containing batch_size samples'
        X = []
        y = []

        for filename in file_path_list_temp:
            # Carga de imagen y extraccion de mascara
            if filename[-4:] == ".png":
                image = (plt.imread(self.dir_x + filename)*255).astype(np.uint8)
                image_masked = plt.imread(self.dir_y + filename.strip(".png") + "-MASK.png")
            else:
                image = plt.imread(self.dir_x + filename).astype(np.uint8)
                image_masked = plt.imread(self.dir_y + filename.strip(".tif") + "-MASK.tif") / 255
            if image.shape[2] > 3:
                image[image[:, :, 3] == 0] = [255, 255, 255, 0]
                image = image[:, :, :-1]
            target = np.array(self.target) / 255
            mask = image_masked.copy()
            mask = np.all(np.abs(mask[:, :] - target) < 1e-1, axis=2).astype(float)
            mask = np.expand_dims(mask, 2)

            # Definicion del pipeline de aumentaciones
            augmentations_train = alb.Compose([
                alb.CLAHE(p=0.1),
                alb.Blur(p=0.1),
                alb.RandomBrightnessContrast(p=0.1, brightness_limit=0.2, contrast_limit=0.2),
                alb.Rotate(p=0.1, limit=5),
                alb.ToFloat(max_value=255)
            ])

            augmentations_val = alb.Compose([
                alb.ToFloat(max_value=255)
            ])

            # Aumentacion
            if self.train:
                augmented = augmentations_train(image=image, mask=mask)
            else:
                augmented = augmentations_val(image=image, mask=mask)

            image = augmented['image']
            if self.color_space == "lab":
                image = color.rgb2lab(image)
                image[:, :, 0] = image[:, :, 0] / 100
                image[:, :, 1] = image[:, :, 1] / 128
                image[:, :, 2] = image[:, :, 2] / 128
            X.append(image)
            y.append(augmented['mask'])

        X = np.array(X).reshape(self.batch_size, self.img_size, self.img_size, 3)
        y = np.array(y).reshape(self.batch_size, self.img_size, self.img_size, 1)

        return X, y

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result