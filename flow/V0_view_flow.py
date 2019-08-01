import pixelhouse as ph
import numpy as np
import glob


F_FLOW = sorted(glob.glob("data/flows/*"))
F_IMG = sorted(glob.glob("data/frames/*"))

for f0, f1 in zip(F_IMG, F_FLOW):

    flow = np.load(f1)
    c0 = ph.load(f0).resize(output_size=(flow.shape[:2][::-1]))

    
    #flow -= flow.min()
    #flow/= flow.max()

    flow += 3
    flow /= 6
    print(flow.min(), flow.max())
    flow = np.clip(flow, 0, 1)
        
    c1 = c0.blank()
    c1.img[:, :, 0] = (flow[:, :, 0]*255).astype(np.uint8)
    c1.img[:, :, 2] = (flow[:, :, 1]*255).astype(np.uint8)
    c1 += ph.filters.gaussian_blur()
    c1.show()
    #c0.show()


