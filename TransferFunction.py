from decimal import Decimal, ROUND_HALF_UP

def transferFunction(lista):
    percentage = 0.35
    numberImagesToAugment = int(Decimal(len(lista) * (1 - percentage)).quantize(0, ROUND_HALF_UP))
    return lista[0:numberImagesToAugment]