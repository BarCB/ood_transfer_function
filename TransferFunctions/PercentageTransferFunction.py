from typing import List
from TransferFunctions.TransferFunction import TransferFunction

class PercentageTransferFunction(TransferFunction):
    """
        Transfer function based on percentage value, marks the images that enters the percentage of the best scores
        :percentage: value between 0 and 1
        :inverse_transfer_function: marks the images outside of the percentage, the images with the worst score
        :return: score value determined to be the threshold 
    """
    def __init__(self, percentage:float, inverse_transfer_function:bool) -> None:
        self.percentage = percentage
        self.inverse_transfer_function = inverse_transfer_function
        super().__init__()

    def filter_batch(self, images_score:List[float]) -> List[float]:
        threshold = self.get_threshold(images_score)
        print("Threshold for the batch: ", threshold)
        images_to_augmentate = []
        for image_index in range(len(images_score)):
            if ((not self.inverse_transfer_function) and images_score[image_index] <= threshold) or (self.inverse_transfer_function and images_score[image_index] > threshold):
                images_to_augmentate.append(1)
            else:
                images_to_augmentate.append(0)

        return images_to_augmentate

    def  get_threshold(self, score_per_image):
        """
        Gets the threshold value that marks the limits between 'self.percentage' and (1 - percentage) of scores
        :return: score value determined to be the threshold 
        """
        copied_list = score_per_image.copy()
        copied_list.sort()
        num_to_filter = int(len(copied_list) * self.percentage)
        threshold = copied_list[num_to_filter]
        return threshold

