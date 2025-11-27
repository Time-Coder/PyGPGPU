from .genVec import genVec


class genVec16(genVec):
    
    swizzle_map = {
        'x': 0, 'y': 1, 'z': 2, 'w': 3,
        'r': 0, 'g': 1, 'b': 2, 'a': 3,
        '0': 0, '1': 1, '2': 2, '3': 3,
        '4': 4, '5': 5, '6': 6, '7': 7,
        '8': 8, '9': 9, 'A': 10, 'B': 11,
        'C': 12, 'D': 13, 'E': 14, 'F': 15,
        'a': 10, 'b': 11,
        'c': 12, 'd': 13, 'e': 14, 'f': 15
    }

    def __len__(self)->int:
        return 16