from .genVec import genVec


class genVec3(genVec):
    
    swizzle_map = {
        'x': 0, 'y': 1, 'z': 2,
        'r': 0, 'g': 1, 'b': 2,
        '0': 0, '1': 1, '2': 2
    }

    def __len__(self)->int:
        return 3