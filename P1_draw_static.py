from tqdm import tqdm
import h5py, os
import matplotlib
import numpy as np
from functools import partial

#circle_radius = 0.001
circle_radius = 0.0005


save_dest = 'ph_figures'
os.system(f'mkdir -p {save_dest}')
f_save = os.path.join('data/point_clouds/', 'points.h5')

with h5py.File(f_save, 'r') as h5:
    
    #points = h5['points'][...]
    points = h5['points'][:, :, :]
    print(points.shape)

    base_points = h5['base_points'][...]
    t = h5['t'][...]

##################################################################

import pixelhouse as ph
pal = ph.palette(0)

bg_color = np.array(pal[0])/40
print(bg_color)
#bg_color = 'k'

canvas = ph.Canvas(width=1280, height=1280, bg=bg_color, extent=3.0)

# Identify the pairs
print(base_points.shape)
yp = base_points[0][1]
idx = np.argsort(np.argsort(yp))
c0 = np.where(np.isin(idx,[0,5]))[0]
c1 = np.where(np.isin(idx,[1,4]))[0]
c2 = np.where(np.isin(idx,[2,3]))[0]
cidx = np.array([c0[0], c0[1], c1[0], c1[1], c2[0], c2[1]])
frame = 0

prior_bp = None
from scipy.spatial.distance import cdist
#from pycpd import affine_registration,deformable_registration

#def point_cloud_affine(iteration, error, X, Y):
#    print("HERE", iteration, error, X, Y)
#    print("EXXX\n",X)
#callback = partial(point_cloud_affine)

for bp, pt in tqdm(zip(base_points, points)):
    
#for bp, pt in zip(base_points, points):

    '''
    if prior_bp is not None:
        #help(deformable_registration)
        reg = deformable_registration(X=prior_bp, Y=bp)
        reg.register(callback)
        #print(reg)      
        #print("HI")
        #print(dir(reg))
        #print(reg.D, reg.G)
        #print(reg.M, reg.N)
        #print(reg.Np, reg.P)
        #print(reg.P1, reg.Pt1)
        #print(reg.TY.shape, reg.W.shape)
        #print(reg.X.shape, reg.Y.shape)
        #print(prior_bp)
        #print(bp)
        #print((bp-prior_bp).round(2))
        #print(bp.round(2))
        #print(prior_bp.round(2))
        #print((reg.W-bp).round(2))
        #print(reg.W.round(3))
        
        #exit()
        
        
        #exit()
        dist = cdist(prior_bp.T, bp.T)
        #np.fill_diagonal(dist, 100)

        idx = []
        for i, row in enumerate(dist):
            j = None
            row = row.tolist()
            while j is None or j in idx:
                j = np.argmin(row)
            

        
        idx = np.argmin(dist, axis=0)
        cidx = cidx[idx]
        #print(dist)
        #print(dist.shape)
        print(idx)      
        #exit()
    '''
    

    BP = np.array([

        [bp[0][cidx[0]], bp[1][cidx[0]]],
        [bp[0][cidx[1]], bp[1][cidx[1]]],
        
        [bp[0][cidx[2]], bp[1][cidx[2]]],
        [bp[0][cidx[3]], bp[1][cidx[3]]],

        [bp[0][cidx[4]], bp[1][cidx[4]]],
        [bp[0][cidx[5]], bp[1][cidx[5]]],

    ])

    

    COLORS = [
        pal[1], pal[1],
        pal[2], pal[2],
        pal[3], pal[3],
    ]

    COLORS = [pal[1],]*6
    

    '''
    for _ in range(20):
        canvas += ph.filters.gaussian_blur(0.5, 0.5)
    '''
    
    # Draw the centroids as a light blurry dus
    for (x,y), c in zip(BP, COLORS):
       canvas += ph.circle(x, y, r=0.025, color=pal[4])
    canvas += ph.filters.gaussian_blur(0.4, 0.4)
    for (x,y), c in zip(BP, COLORS):
        canvas += ph.circle(x, y, r=0.025, color=pal[4])

    
    def draw_color_points(canvas):
        for q in pt.T[:]:
            w = np.exp(-2*np.linalg.norm(BP-q,axis=1))
            w /= w.sum()
            w = (w.reshape(-1,1)*COLORS).sum(axis=0)
            c = np.clip(w, 0, 255).astype(np.uint8)

            px = canvas.transform_x(q[0])
            py = canvas.transform_y(q[1])

            if px >= canvas.shape[0]:
                continue

            if py >= canvas.shape[1]:
                continue

            if(px<0 or py<0):
                continue
            
            canvas.img[px, py] = c

            #canvas.img[px+1, py] = c
            #canvas.img[px-1, py+1] = c

            #canvas += ph.circle(q[0], q[1], r=circle_radius, color=c.tolist())

    
    #print(canvas.transform_kernel_length(0.02))
    #print(canvas.transform_kernel_length(0.0000000001))
    #exit()
    
    with canvas.layer() as C2:
        draw_color_points(C2)
        #C2 += ph.filters.gaussian_blur(0.02, 0.02)

        #with C2.layer() as C3:
        #    draw_color_points(C3)
        #    #print(C3)
        #    #exit()
            

    canvas.img = canvas.img[128*2:-128*2, :]
    
    print(canvas.shape)
    canvas.save('demo.png')
    canvas.resize(0.5).show()

    exit()
    
    #if frame < 30:
    canvas.save(f'ph_figures/{frame:03d}.png')
    frame += 1

    #canvas.show(1)
    canvas.show()
    exit()
    canvas = canvas.blank()
    prior_bp = bp

