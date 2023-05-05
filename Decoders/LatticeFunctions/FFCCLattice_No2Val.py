import numpy as np

import matplotlib.pyplot as plt


########################################
########## FFCC Lattice Class  ##########
########################################

class FFCCLattice_No2Val(object):
    def __init__(self, x_size, y_size=None, z_size=None):
        ## boundary is considered to be periodic for simplicity

        self.lattice_x_size = x_size
        if y_size is not None:
            self.lattice_y_size = y_size
        else:
            self.lattice_y_size = x_size
        if z_size is not None:
            self.lattice_z_size = z_size
        else:
            self.lattice_z_size = x_size

        self.num_cells = self.lattice_x_size * self.lattice_y_size * self.lattice_z_size
        self.data_type, self.empty_ix = self.get_data_type()

        # 3d spatial shifts to move between lattice cells. Defines the lattice cartesian system
        self.cell_xshift = np.array((1, 0, 0))
        self.cell_yshift = np.array((0.5, 0.5*np.sqrt(3.)/2., 0))
        self.cell_zshift = np.array((0, 0, 1))

        # Shift to move between qubits in the lattice.
        # These correspond to the half length of exagon sides (in x and y) and to distance between layers of the code.
        self.xshift = self.cell_xshift / 6
        self.yshift = self.cell_yshift / 6
        self.zshift = self.cell_zshift / 6  # z shift between different layers of the code.

        self.primal_qbts_positions = {}
        self.dual_qbts_positions = {}
        self.edged_qbts = []  # list of lattice edges as [(primal qbt1, dual qbt1), (primal qbt2, dual qbt2), ...]

        self.cells_shapes = []
        self.syns_shapes = []

        self.num_primal_qbts = 0
        self.num_dual_qbts = 0
        self.tot_num_qbts = 0

        self.current_primal_qbt_ix = 0

        self.current_dual_qbt_ix = 0

        # This is structured as:
        # [[[primal qbts in cell 1],[dual qbts in cell 1]], [[primal qbts in cell 2],[dual qbts in cell 2]], ... ]
        self.cells_qbts_struct = np.array([[[self.empty_ix] * 27]*2]*self.num_cells, dtype=self.data_type)

        # This is structured as:
        # [[[qbts in red primal synd cell1], [qbts in green primal synd cell1], [qbts in blue primal synd cell1]],
        #  [[qbts in red primal synd cell2], [qbts in green primal synd cell2], [qbts in blue primal synd cell2]], ...]
        # The syndromes in each cell are arranged as:
        #                                                    /
        #                                               ____/  blue
        #                                              /    \
        #                                             /  red \_____
        #                                             \      /
        #                                              \____/  green
        #                                                   \
        #                                                    \
        self.cells_primal_syndrs_struct = np.array([[[self.empty_ix]*18]*3]*self.num_cells, dtype=self.data_type)

        self.log_ops_qbts = [None, None]  # logical ops for x-like and z-like surfaces

        self.build_lattice_qubits()
        self.build_lattice_edges()
        self.build_syndromes()
        self.get_logical_operators()

    def build_lattice_qubits(self):
        # Starts building the lattice
        for z_ix in range(self.lattice_z_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):
                    cell_pos = (x_ix + 0.5) * self.cell_xshift + (y_ix + 0.5) * self.cell_yshift + \
                               (z_ix + 0.5) * self.cell_zshift
                    cell_ix = self.cell_xyzcoords_to_ix([x_ix, y_ix, z_ix])

                    # Add primal qubits
                    # Layer 0Red
                    qbt_pos = cell_pos + 2*self.xshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][0] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.xshift - 2*self.yshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][1] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.yshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][2] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][3] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 2*self.yshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][4] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.yshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][5] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    # Layer 2Blue
                    qbt_pos = cell_pos + 2*self.xshift + 2*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][9] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.xshift - 2*self.yshift + 2*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][10] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.yshift + 2*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][11] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 2*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][12] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 2*self.yshift + 2*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][13] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.yshift + 2*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][14] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1


                    # Layer 4Green
                    qbt_pos = cell_pos + 2*self.xshift + 4*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][18] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.xshift - 2*self.yshift + 4*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][19] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.yshift + 4*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][20] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 4*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][21] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 2*self.yshift + 4*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][22] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.yshift + 4*self.zshift
                    self.primal_qbts_positions[self.current_primal_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 0][23] = self.current_primal_qbt_ix
                    self.current_primal_qbt_ix += 1



                    # Add dual qubits

                    # Layer 1Green
                    qbt_pos = cell_pos + 2*self.xshift + self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][3] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.xshift - 2*self.yshift + self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][4] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.yshift + self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][5] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][6] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 2*self.yshift + self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][7] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.yshift + self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][8] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    # Layer 3Red
                    qbt_pos = cell_pos + 2*self.xshift + 3*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][12] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.xshift - 2*self.yshift + 3*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][13] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.yshift + 3*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][14] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 3*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][15] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 2*self.yshift + 3*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][16] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.yshift + 3*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][17] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    # Layer 5Blue
                    qbt_pos = cell_pos + 2*self.xshift + 5*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][21] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.xshift - 2*self.yshift + 5*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][22] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.yshift + 5*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][23] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 5*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][24] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos - 2*self.xshift + 2*self.yshift + 5*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][25] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1

                    qbt_pos = cell_pos + 2*self.yshift + 5*self.zshift
                    self.dual_qbts_positions[self.current_dual_qbt_ix] = qbt_pos
                    self.cells_qbts_struct[cell_ix, 1][26] = self.current_dual_qbt_ix
                    self.current_dual_qbt_ix += 1
        self.num_primal_qbts = len(self.primal_qbts_positions)
        self.num_dual_qbts = len(self.dual_qbts_positions)

    def build_syndromes(self):
        ## TODO: Now it builds only primal syndromes, extend to duals
        for cell_ix, all_cell_qbts in enumerate(self.cells_qbts_struct):
            primal_qbts = all_cell_qbts[0]

            # Red syndrome
            self.cells_primal_syndrs_struct[cell_ix][0][0] = primal_qbts[6]
            self.cells_primal_syndrs_struct[cell_ix][0][1] = primal_qbts[7]
            self.cells_primal_syndrs_struct[cell_ix][0][2] = primal_qbts[8]
            self.cells_primal_syndrs_struct[cell_ix][0][3] = primal_qbts[9]
            self.cells_primal_syndrs_struct[cell_ix][0][4] = primal_qbts[10]
            self.cells_primal_syndrs_struct[cell_ix][0][5] = primal_qbts[11]
            self.cells_primal_syndrs_struct[cell_ix][0][6] = primal_qbts[12]
            self.cells_primal_syndrs_struct[cell_ix][0][7] = primal_qbts[13]
            self.cells_primal_syndrs_struct[cell_ix][0][8] = primal_qbts[14]
            self.cells_primal_syndrs_struct[cell_ix][0][9] = primal_qbts[18]
            self.cells_primal_syndrs_struct[cell_ix][0][10] = primal_qbts[19]
            self.cells_primal_syndrs_struct[cell_ix][0][11] = primal_qbts[20]
            self.cells_primal_syndrs_struct[cell_ix][0][12] = primal_qbts[21]
            self.cells_primal_syndrs_struct[cell_ix][0][13] = primal_qbts[22]
            self.cells_primal_syndrs_struct[cell_ix][0][14] = primal_qbts[23]
            self.cells_primal_syndrs_struct[cell_ix][0][15] = primal_qbts[24]
            self.cells_primal_syndrs_struct[cell_ix][0][16] = primal_qbts[25]
            self.cells_primal_syndrs_struct[cell_ix][0][17] = primal_qbts[26]

            # Green syndrome
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][1][0] = self.cells_qbts_struct[neigh_cell][0][25]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][1][1] = self.cells_qbts_struct[neigh_cell][0][26]

            neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][1][2] = self.cells_qbts_struct[neigh_cell][0][24]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][3] = self.cells_qbts_struct[neigh_cell][0][2]
            self.cells_primal_syndrs_struct[cell_ix][1][8] = self.cells_qbts_struct[neigh_cell][0][3]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][4] = self.cells_qbts_struct[neigh_cell][0][5]
            self.cells_primal_syndrs_struct[cell_ix][1][5] = self.cells_qbts_struct[neigh_cell][0][4]

            self.cells_primal_syndrs_struct[cell_ix][1][6] = primal_qbts[1]
            self.cells_primal_syndrs_struct[cell_ix][1][7] = primal_qbts[0]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][9] = self.cells_qbts_struct[neigh_cell][0][11]
            self.cells_primal_syndrs_struct[cell_ix][1][14] = self.cells_qbts_struct[neigh_cell][0][12]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][10] = self.cells_qbts_struct[neigh_cell][0][14]
            self.cells_primal_syndrs_struct[cell_ix][1][11] = self.cells_qbts_struct[neigh_cell][0][13]

            self.cells_primal_syndrs_struct[cell_ix][1][12] = primal_qbts[10]
            self.cells_primal_syndrs_struct[cell_ix][1][13] = primal_qbts[9]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][15] = self.cells_qbts_struct[neigh_cell][0][17]

            self.cells_primal_syndrs_struct[cell_ix][1][16] = primal_qbts[16]
            self.cells_primal_syndrs_struct[cell_ix][1][17] = primal_qbts[15]


            # Blue syndrome
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][0] = self.cells_qbts_struct[neigh_cell][0][16]

            self.cells_primal_syndrs_struct[cell_ix][2][1] = primal_qbts[15]
            self.cells_primal_syndrs_struct[cell_ix][2][2] = primal_qbts[17]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][3] = self.cells_qbts_struct[neigh_cell][0][22]
            self.cells_primal_syndrs_struct[cell_ix][2][4] = self.cells_qbts_struct[neigh_cell][0][21]

            self.cells_primal_syndrs_struct[cell_ix][2][5] = primal_qbts[18]
            self.cells_primal_syndrs_struct[cell_ix][2][6] = primal_qbts[23]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][7] = self.cells_qbts_struct[neigh_cell][0][20]
            self.cells_primal_syndrs_struct[cell_ix][2][8] = self.cells_qbts_struct[neigh_cell][0][19]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][9] = self.cells_qbts_struct[neigh_cell][0][4]
            self.cells_primal_syndrs_struct[cell_ix][2][10] = self.cells_qbts_struct[neigh_cell][0][3]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][11] = self.cells_qbts_struct[neigh_cell][0][0]
            self.cells_primal_syndrs_struct[cell_ix][2][12] = self.cells_qbts_struct[neigh_cell][0][5]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][13] = self.cells_qbts_struct[neigh_cell][0][2]
            self.cells_primal_syndrs_struct[cell_ix][2][14] = self.cells_qbts_struct[neigh_cell][0][1]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][15] = self.cells_qbts_struct[neigh_cell][0][8]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][16] = self.cells_qbts_struct[neigh_cell][0][6]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][17] = self.cells_qbts_struct[neigh_cell][0][7]

    # Function to calculate the logical operators of the lattice
    def get_logical_operators(self):
        log_x = []
        log_y = []
        log_z = []

        # builds logical operator surface on z direction
        z_ix = 0
        for y_ix in range(self.lattice_y_size):
            for x_ix in range(self.lattice_x_size):
                cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                primal_qubits = self.cells_qbts_struct[cell_ix][0]
                log_z += list(primal_qubits[[0, 1, 2, 3, 4, 5]])

        self.log_ops_qbts = [log_x, log_y, log_z]


    # Functions that build the matching matrix for syndromes of the lattice
    def get_matching_matrix(self):
        h_matrix = np.zeros((3*self.num_cells, self.num_primal_qbts), dtype=np.uint8)
        for cell_ix, cell_syndrs in enumerate(self.cells_primal_syndrs_struct):
            for syndr_ix, syndr_nodes_list in enumerate(cell_syndrs):
                syndr_qubits = syndr_nodes_list[syndr_nodes_list != self.empty_ix]
                h_matrix[3*cell_ix + syndr_ix, syndr_qubits] = np.ones(len(syndr_qubits), dtype=np.uint8)
        return h_matrix

    # Functions to deal and move in between with cells in the lattice

    def cell_xyzcoords_to_ix(self, cell_xytcoords):
        return cell_xytcoords[0] + cell_xytcoords[1] * self.lattice_x_size + \
               cell_xytcoords[2] * self.lattice_x_size * self.lattice_y_size

    def cell_ix_to_xyzcoords(self, cell_ix):
        t_coord = int(cell_ix / (self.lattice_x_size * self.lattice_y_size))
        y_coord = int(cell_ix / self.lattice_x_size) % self.lattice_y_size
        x_coord = cell_ix % self.lattice_x_size
        return np.array((x_coord, y_coord, t_coord))

    def shifted_cell_ix(self, cell_ix, shift, shift_axis):
        #     Function to get the index of a cell obtained shifting an initial cell with 'cell_ix'
        #     by a integer (positive or negative) step 'shift' along an axis 'shift_axis' in x, y, or t.
        if not isinstance(shift, int):
            raise ValueError('The parameter shift can only be an integer (positive or negative)')

        if shift_axis == 'x':
            axis_label = 0
            size_lim = self.lattice_x_size
        elif shift_axis == 'y':
            axis_label = 1
            size_lim = self.lattice_y_size
        elif shift_axis == 'z':
            axis_label = 2
            size_lim = self.lattice_z_size
        else:
            raise ValueError('Shift axis can only be one of (x, y, or z)')
        temp_coords = self.cell_ix_to_xyzcoords(cell_ix)
        temp_coords[axis_label] = (temp_coords[axis_label] + shift) % size_lim
        return self.cell_xyzcoords_to_ix(temp_coords)


    # Data type handler
    def get_data_type(self):
        est_num_qbts_log = np.log2(54 * self.num_cells)
        if est_num_qbts_log < 8:
            data_type = np.uint8
        elif est_num_qbts_log < 16:
            data_type = np.uint16
        elif est_num_qbts_log < 32:
            data_type = np.uint32
        else:
            data_type = np.uint64
        return data_type, np.iinfo(data_type).max


    # Functions for drawing the lattice
    def draw_lattice(self, print_nodes=True, print_edges=True, print_cells=False, primal_errors=None,
                    fig_size=10, fig_aspect=None):
        if fig_aspect is None:
            fig_aspect = self.lattice_z_size / ((self.lattice_x_size + self.lattice_y_size) / 1)

        node_size = 100 / (self.lattice_x_size * self.lattice_y_size)
        primal_node_color = 'blue'
        dual_node_color = 'black'
        error_node_color = 'red'

        edge_color = 'black'
        edge_alpha = 1
        edge_width = 1

        cell_color = 'black'
        cell_alpha = 0.5
        cell_width = 2
        cell_style = ':'

        fig = plt.figure(figsize=(fig_size, fig_aspect * fig_size))

        ax = fig.add_subplot(projection='3d')

        if primal_errors is not None:
            qbts_with_errors = np.where(primal_errors)[0]
            if len(qbts_with_errors) > 0:
                primal_errors_pos_array = np.array([self.primal_qbts_positions[err_qbt_ix]
                                                    for err_qbt_ix in qbts_with_errors])
                ax.scatter(
                    primal_errors_pos_array[:, 0], primal_errors_pos_array[:, 1], primal_errors_pos_array[:, 2],
                    s=1.2*node_size, c=error_node_color, alpha=1)

        if print_nodes:
            primal_pos_array = np.array(list(self.primal_qbts_positions.values()))
            dual_pos_array = np.array(list(self.dual_qbts_positions.values()))
            ax.scatter(
                primal_pos_array[:, 0], primal_pos_array[:, 1], primal_pos_array[:, 2],
                s=node_size, c=primal_node_color)
            ax.scatter(
                dual_pos_array[:, 0], dual_pos_array[:, 1], dual_pos_array[:, 2],
                s=node_size, c=dual_node_color)

        if print_edges:
            plotted_lattice_edges = \
                np.array([np.array((self.primal_qbts_positions[primal_qbt], self.dual_qbts_positions[dual_qbt]))
                          for primal_qbt, dual_qbt in self.edged_qbts])
            # self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.yshift)))
            for edge_plot in plotted_lattice_edges:
                ax.plot3D(*edge_plot.T, edge_color, alpha=edge_alpha, lw=edge_width)

        if print_cells:
            plotted_cells_edges = []
            for cell_struct in self.cells_qbts_struct:
        # # [[[primal qbts in cell 1],[dual qbts in cell 1]], [[primal qbts in cell 2],[dual qbts in cell 2]], ... ]
        # self.cells_qbts_struct = np.array([[[self.empty_ix] * 27]*2]*self.num_cells, dtype=self.data_type)
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][0]],
                              self.primal_qbts_positions[cell_struct[0][1]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][1]],
                              self.primal_qbts_positions[cell_struct[0][2]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][2]],
                              self.primal_qbts_positions[cell_struct[0][3]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][3]],
                              self.primal_qbts_positions[cell_struct[0][4]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][4]],
                              self.primal_qbts_positions[cell_struct[0][5]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][5]],
                              self.primal_qbts_positions[cell_struct[0][0]])))

                plotted_cells_edges.append(
                    np.array((self.dual_qbts_positions[cell_struct[1][21]],
                              self.dual_qbts_positions[cell_struct[1][22]])))
                plotted_cells_edges.append(
                    np.array((self.dual_qbts_positions[cell_struct[1][22]],
                              self.dual_qbts_positions[cell_struct[1][23]])))
                plotted_cells_edges.append(
                    np.array((self.dual_qbts_positions[cell_struct[1][23]],
                              self.dual_qbts_positions[cell_struct[1][24]])))
                plotted_cells_edges.append(
                    np.array((self.dual_qbts_positions[cell_struct[1][24]],
                              self.dual_qbts_positions[cell_struct[1][25]])))
                plotted_cells_edges.append(
                    np.array((self.dual_qbts_positions[cell_struct[1][25]],
                              self.dual_qbts_positions[cell_struct[1][26]])))
                plotted_cells_edges.append(
                    np.array((self.dual_qbts_positions[cell_struct[1][26]],
                              self.dual_qbts_positions[cell_struct[1][21]])))


                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][0]],
                              self.dual_qbts_positions[cell_struct[1][21]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][1]],
                              self.dual_qbts_positions[cell_struct[1][22]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][2]],
                              self.dual_qbts_positions[cell_struct[1][23]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][3]],
                              self.dual_qbts_positions[cell_struct[1][24]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][4]],
                              self.dual_qbts_positions[cell_struct[1][25]])))
                plotted_cells_edges.append(
                    np.array((self.primal_qbts_positions[cell_struct[0][5]],
                              self.dual_qbts_positions[cell_struct[1][26]])))

            for cell_edge_plot in plotted_cells_edges:
                ax.plot3D(*cell_edge_plot.T, cell_color, alpha=cell_alpha, lw=cell_width, ls=cell_style)


        plt.axis('off')
        plt.show()
