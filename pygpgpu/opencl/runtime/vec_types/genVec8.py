from .genVec import genVec


class genVec8(genVec):
    
    swizzle_map = {
        'x': 0, 'y': 1, 'z': 2, 'w': 3,
        'r': 0, 'g': 1, 'b': 2, 'a': 3,
        '0': 0, '1': 1, '2': 2, '3': 3,
        '4': 4, '5': 5, '6': 6, '7': 7
    }

    def __len__(self)->int:
        return 8