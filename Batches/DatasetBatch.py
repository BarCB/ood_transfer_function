class DatasetBatch:
    def __init__(self, images, labels, size:int) -> None:
        self.images = images
        self.labels = labels
        self.size = size

    def getImages(self):
        return self.images