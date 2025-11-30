from .genVec import genVec


class genVec2(genVec):

    swizzle_map = {
        'x': 0, 'y': 1,
        'r': 0, 'g': 1,
        '0': 0, '1': 1
    }

    def __len__(self)->int:
        return 2