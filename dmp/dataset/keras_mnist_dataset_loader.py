from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

from dmp.dataset.keras_image_dataset_loader import KerasImageDatasetLoader


class KerasMNISTDatasetLoader(KerasImageDatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        keras_load_data_function: Callable,
    ) -> None:
        super().__init__(dataset_name, keras_load_data_function)

    # def __call__(self):
    #     result = super().__call__()
    #     # example of loading the mnist dataset
    #     # from keras.datasets import mnist
    #     from matplotlib import pyplot
    #     # load dataset
    #     # (trainX, trainy), (testX, testy) = mnist.load_data()
    #     # summarize loaded dataset
    #     # print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    #     # print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
    #     # plot first few images
    #     for i in range(9):
    #         # define subplot
    #         pyplot.subplot(330 + 1 + i)
    #         # plot raw pixel data
    #         pyplot.imshow(result.train.inputs[i], cmap=pyplot.get_cmap('gray'))
    #         # show the figure
    #         pyplot.show()
    #     return result

    def _prepare_inputs(self, data):
        return super()._prepare_inputs(data.reshape(*data.shape, 1))
