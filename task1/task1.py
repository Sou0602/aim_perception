import numpy as np
import argparse

def get_cone_mask(coord_1x,coord_1y,orientation,fov,len_grid):
    """ Computes a 2D cone for each police position given its orientation, field of view and position"""

    if fov == 360:
        return np.ones((len_grid, len_grid), dtype=np.int16)
    elif fov == 0:
        return np.zeros((len_grid,len_grid),dtype=np.int16)

    else:
        m1u = np.tan((orientation - fov/2)*np.pi/180)
        m1d = np.tan((orientation + fov/2)*np.pi/180)

        ## Four different possible notations to represent the grid. Checking for all four
        g1x = np.linspace(-0.5,len_grid-1.5,len_grid)
        g1y = np.linspace(-0.5,len_grid-1.5,len_grid)
        g2x = np.linspace(0.5, len_grid-0.5,len_grid)
        g2y = np.linspace(-0.5,len_grid-1.5,len_grid)
        g3x = np.linspace(-0.5,len_grid-1.5,len_grid)
        g3y = np.linspace(0.5,len_grid-0.5,len_grid)
        g4x = np.linspace(0.5,len_grid-0.5,len_grid)
        g4y = np.linspace(0.5,len_grid-0.5,len_grid)

        ## Interior point
        if orientation == 90:
            point_in_x = coord_1x
            point_in_y = coord_1y - 0.05
        elif orientation == 270:
            point_in_x = coord_1x
            point_in_y = coord_1y + 0.05
        else:
            m_or = np.tan(orientation*np.pi/180)
            point_in_x = coord_1x - 0.05 ##small displacement
            point_in_y  = coord_1y  - m_or*(point_in_x - coord_1x) ## Flip axis direction notation

        lui = point_in_y + m1u*point_in_x - (coord_1y + m1u*coord_1x)
        if lui < 0:
            sign = 1
        else:
            sign = -1

        mask1u = (g1y[:,np.newaxis] + m1u* g1x[np.newaxis,:] - (coord_1y + m1u*coord_1x))*sign < 0
        mask2u = (g2y[:,np.newaxis] + m1u* g2x[np.newaxis,:] - (coord_1y + m1u*coord_1x))*sign < 0
        mask3u = (g3y[:,np.newaxis] + m1u* g3x[np.newaxis,:] - (coord_1y + m1u*coord_1x))*sign < 0
        mask4u = (g4y[:,np.newaxis] + m1u* g4x[np.newaxis,:] - (coord_1y + m1u*coord_1x))*sign < 0

        masku = np.array(mask1u) + np.array(mask2u) + np.array(mask3u) + np.array(mask4u)

        ldi = point_in_y + m1d*point_in_x - (coord_1y + m1d*coord_1x)
        if ldi < 0:
            sign = 1
        else:
            sign = -1


        mask1 = (g1y[:,np.newaxis] + m1d* g1x[np.newaxis,:] - (coord_1y + m1d*coord_1x))*sign < 0
        mask2 = (g2y[:,np.newaxis] + m1d* g2x[np.newaxis,:] - (coord_1y + m1d*coord_1x))*sign < 0
        mask3 = (g3y[:,np.newaxis] + m1d* g3x[np.newaxis,:] - (coord_1y + m1d*coord_1x))*sign < 0
        mask4 = (g4y[:,np.newaxis] + m1d* g4x[np.newaxis,:] - (coord_1y + m1d*coord_1x))*sign < 0

        maskd = np.array(mask1) + np.array(mask2) + np.array(mask3) + np.array(mask4)

        maskd_i = np.array(maskd,dtype=np.int16)
        masku_i = np.array(masku,dtype=np.int16)

        ## If fov<180
        ## For all four cases of coordinate notations
        ## Constraining the cone to only see the front part and not the back part
        if fov < 180:
            if orientation == 180:
                mask_or = g1x[np.newaxis,:] - coord_1x < 0
            elif orientation == 0 or orientation == 360:
                mask_or = g1x[np.newaxis,:] - coord_1x > 0
            elif orientation == 90:
                mask_or = g1y[:,np.newaxis] - coord_1y < 0
            elif orientation == 270:
                mask_or = g1y[:, np.newaxis] - coord_1y> 0
            else:
                m_nor = (-1/m_or)
                lni = point_in_y + m_nor * point_in_x - (coord_1y + m_nor * coord_1x)
                if lni  < 0:
                    sign = 1
                else:
                    sign = -1
                mask_or_1 = (g1y[:, np.newaxis] + m_nor * g1x[np.newaxis, :] - (coord_1y + m_nor * coord_1x)) * sign < 0
                mask_or_2 = (g2y[:, np.newaxis] + m_nor * g2x[np.newaxis, :] - (coord_1y + m_nor * coord_1x)) * sign < 0
                mask_or_3 = (g3y[:, np.newaxis] + m_nor * g3x[np.newaxis, :] - (coord_1y + m_nor * coord_1x)) * sign < 0
                mask_or_4 = (g4y[:, np.newaxis] + m_nor * g4x[np.newaxis, :] - (coord_1y + m_nor * coord_1x)) * sign < 0

                mask_or = np.array(mask_or_1) + np.array(mask_or_2) + np.array(mask_or_3) + np.array(mask_or_4)

            mask_or = np.array(mask_or,dtype=np.int16)
            mask_cone = maskd_i * masku_i * mask_or
        else:
            mask_cone = np.array((maskd_i + masku_i) > 0,dtype=np.int16)
        return mask_cone


def thief_and_cops(grid,orientations,fov):
    """Main function that counts the number of police and computes a field of view 2D cone for each police.
       The masks are combined to find the cells outside the field of view for the thief to escape."""

    assert len(orientations) == len(fov)
    coord_T = np.where(np.array(grid) == -1)
    grid_size = grid.shape[0]
    num_police = np.max(grid)
    assert num_police == len(orientations)

    coords_p = []
    mask_cones = []
    for k in range(1,num_police+1):
        coords_p.append(np.where(grid==k))
        mask = get_cone_mask(coords_p[-1][1],coords_p[-1][0],orientations[k-1],fov[k-1],grid_size)
        mask_cones.append(mask)

    mask_police = mask_cones[0]
    if len(mask_cones) > 1:
        mask_police = np.array(sum(mask_cones) > 0,dtype=np.int16)
    empty_mask = 1 - mask_police ## empty_mask 1 indicates a position out of field of view
    thief_in_police_view = []
    for i in range(len(mask_cones)):
        if mask_cones[i][coord_T] == 1:
            thief_in_police_view.append(i+1)
    if len(thief_in_police_view) == 0:
        return [],[coord_T[0],coord_T[1]] ## Thief does not need to move if in the field of view of the police
    else:
        e_x, e_y = np.where(empty_mask == 1)
        for k in range(len(e_x)):
            dist = np.array(abs(e_x - coord_T[0]) + abs(e_y - coord_T[1]))
            min_ind = np.argmin(dist)
        return thief_in_police_view, [e_x[min_ind],e_y[min_ind]]

def process_inputs(Grid,Orientations,FoV):
    """Processes the inputs from strings to the format required"""

    assert type(Grid) == str
    assert type(Orientations) == str
    assert type(FoV) == str

    Grid = Grid.replace('T','-1').replace('[','').replace(']','').strip().split(',')
    grid_shape = int(np.sqrt(len(Grid)))
    grid = []
    for g in Grid:
        grid.append(int(g))
    grid = np.array(grid).reshape(grid_shape,grid_shape)

    Orientations = Orientations.replace('[','').replace(']','').strip().split(',')
    orientations = [int(o) for o in Orientations]

    FoV = FoV.replace('[','').replace(']','').strip().split(',')
    fov = [int(f) for f in FoV]

    return grid,orientations,fov

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Grid', type=str, default='[[0,0,0,0,0],[T,0,0,0,2],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0]]', help='Grid')
    parser.add_argument('--Orientations', type=str, default='[180,150]', help='Orientations')
    parser.add_argument('--FoV', type=str, default='[60,60]', help='Orientaions')

    args = parser.parse_args()

    grid,orientation,fov = process_inputs(args.Grid,args.Orientations,args.FoV)

    police, position = thief_and_cops(grid,orientation,fov)
    print(police,position)



