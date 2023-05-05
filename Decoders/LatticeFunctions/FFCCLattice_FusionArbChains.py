import numpy as np


######################################################################################
########## FFCC Lattice with fusion (branched chains with arbitrary length) ##########
######################################################################################

class FFCCLattice_FusionsChains(object):

    def __init__(self, x_size, y_size=None, z_size=None, chains_lenght = 4):
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

        if chains_lenght in [4, 6, 8, 14]:
            self.chains_len = chains_lenght
        else:
            raise ValueError('Only implemented values for chains_lenght are [4, 6, 8, 14]')

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

        self.cells_shapes = []
        self.syns_shapes = []

        self.num_primal_qbts = 0
        self.tot_num_qbts = 0

        self.current_primal_qbt_ix = 0

        self.current_dual_qbt_ix = 0

        # This is structured as:
        # [[fusions in cell 1], [fusions in cell 2], ... ]
        self.cells_qbts_struct = np.array([[self.empty_ix] * 36]*self.num_cells, dtype=self.data_type)

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
        self.cells_primal_syndrs_struct = np.array([[[self.empty_ix]*24]*3]*self.num_cells, dtype=self.data_type)

        self.log_ops_qbts = [None, None]  # logical ops for x-like and z-like surfaces

        self.build_lattice_qubits()
        self.build_syndromes()
        self.get_logical_operators()

    def build_lattice_qubits(self):
        # Starts building the lattice
        for z_ix in range(self.lattice_z_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):
                    cell_ix = self.cell_xyzcoords_to_ix([x_ix, y_ix, z_ix])

                    self.cells_qbts_struct[cell_ix] = [self.current_primal_qbt_ix + i for i in range(36)]
                    self.current_primal_qbt_ix += 36

        self.num_primal_qbts = 36 * len(self.cells_qbts_struct)

    def build_syndromes(self):
        ## TODO: Now it builds only primal syndromes, extend to duals
        for cell_ix, primal_qbts in enumerate(self.cells_qbts_struct):

            # Red syndrome
            if self.chains_len == 4:
                self.cells_primal_syndrs_struct[cell_ix][0][0] = primal_qbts[6]
                self.cells_primal_syndrs_struct[cell_ix][0][1] = primal_qbts[7]
                self.cells_primal_syndrs_struct[cell_ix][0][2] = primal_qbts[8]

            self.cells_primal_syndrs_struct[cell_ix][0][3] = primal_qbts[9]
            self.cells_primal_syndrs_struct[cell_ix][0][4] = primal_qbts[10]
            self.cells_primal_syndrs_struct[cell_ix][0][5] = primal_qbts[11]

            if self.chains_len in [4, 6]:
                self.cells_primal_syndrs_struct[cell_ix][0][6] = primal_qbts[12]
                self.cells_primal_syndrs_struct[cell_ix][0][7] = primal_qbts[13]
                self.cells_primal_syndrs_struct[cell_ix][0][8] = primal_qbts[14]

            self.cells_primal_syndrs_struct[cell_ix][0][9] = primal_qbts[15]
            self.cells_primal_syndrs_struct[cell_ix][0][10] = primal_qbts[16]
            self.cells_primal_syndrs_struct[cell_ix][0][11] = primal_qbts[17]
            self.cells_primal_syndrs_struct[cell_ix][0][12] = primal_qbts[21]
            self.cells_primal_syndrs_struct[cell_ix][0][13] = primal_qbts[22]
            self.cells_primal_syndrs_struct[cell_ix][0][14] = primal_qbts[23]

            if self.chains_len in [4, 6]:
                self.cells_primal_syndrs_struct[cell_ix][0][15] = primal_qbts[24]
                self.cells_primal_syndrs_struct[cell_ix][0][16] = primal_qbts[25]
                self.cells_primal_syndrs_struct[cell_ix][0][17] = primal_qbts[26]

            self.cells_primal_syndrs_struct[cell_ix][0][18] = primal_qbts[27]
            self.cells_primal_syndrs_struct[cell_ix][0][19] = primal_qbts[28]
            self.cells_primal_syndrs_struct[cell_ix][0][20] = primal_qbts[29]

            if self.chains_len == 4:
                self.cells_primal_syndrs_struct[cell_ix][0][21] = primal_qbts[30]
                self.cells_primal_syndrs_struct[cell_ix][0][22] = primal_qbts[31]
                self.cells_primal_syndrs_struct[cell_ix][0][23] = primal_qbts[32]


            # Green syndrome

            if self.chains_len == 4:
                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
                neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
                neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'z')
                self.cells_primal_syndrs_struct[cell_ix][1][0] = self.cells_qbts_struct[neigh_cell][32]

                neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
                self.cells_primal_syndrs_struct[cell_ix][1][1] = self.cells_qbts_struct[neigh_cell][30]

                neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
                neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x')
                self.cells_primal_syndrs_struct[cell_ix][1][2] = self.cells_qbts_struct[neigh_cell][31]


            neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][3] = self.cells_qbts_struct[neigh_cell][34]

            neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][4] = self.cells_qbts_struct[neigh_cell][35]

            neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][1][5] = self.cells_qbts_struct[neigh_cell][33]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][6] = self.cells_qbts_struct[neigh_cell][2]

            self.cells_primal_syndrs_struct[cell_ix][1][7] = primal_qbts[1]

            self.cells_primal_syndrs_struct[cell_ix][1][8] = primal_qbts[0]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][9] = self.cells_qbts_struct[neigh_cell][5]

            self.cells_primal_syndrs_struct[cell_ix][1][10] = primal_qbts[3]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][11] = self.cells_qbts_struct[neigh_cell][4]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][12] = self.cells_qbts_struct[neigh_cell][10]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][13] = self.cells_qbts_struct[neigh_cell][11]

            self.cells_primal_syndrs_struct[cell_ix][1][14] = primal_qbts[9]

            if self.chains_len in [4, 6]:
                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
                neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
                self.cells_primal_syndrs_struct[cell_ix][1][15] = self.cells_qbts_struct[neigh_cell][14]

                self.cells_primal_syndrs_struct[cell_ix][1][16] = primal_qbts[12]

                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
                self.cells_primal_syndrs_struct[cell_ix][1][17] = self.cells_qbts_struct[neigh_cell][13]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][18] = self.cells_qbts_struct[neigh_cell][17]

            self.cells_primal_syndrs_struct[cell_ix][1][19] = primal_qbts[15]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][20] = self.cells_qbts_struct[neigh_cell][16]

            if self.chains_len in [4, 8]:
                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
                neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
                self.cells_primal_syndrs_struct[cell_ix][1][21] = self.cells_qbts_struct[neigh_cell][20]

                self.cells_primal_syndrs_struct[cell_ix][1][22] = primal_qbts[19]

                self.cells_primal_syndrs_struct[cell_ix][1][23] = primal_qbts[18]



            # Blue syndrome

            if self.chains_len in [4, 8]:
                self.cells_primal_syndrs_struct[cell_ix][2][0] = primal_qbts[18]

                self.cells_primal_syndrs_struct[cell_ix][2][1] = primal_qbts[20]

                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
                self.cells_primal_syndrs_struct[cell_ix][2][2] = self.cells_qbts_struct[neigh_cell][19]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][3] = self.cells_qbts_struct[neigh_cell][23]

            self.cells_primal_syndrs_struct[cell_ix][2][4] = primal_qbts[21]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][5] = self.cells_qbts_struct[neigh_cell][22]

            if self.chains_len in [4, 6]:
                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
                self.cells_primal_syndrs_struct[cell_ix][2][6] = self.cells_qbts_struct[neigh_cell][25]

                self.cells_primal_syndrs_struct[cell_ix][2][7] = primal_qbts[26]

                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
                self.cells_primal_syndrs_struct[cell_ix][2][8] = self.cells_qbts_struct[neigh_cell][24]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][9] = self.cells_qbts_struct[neigh_cell][28]

            self.cells_primal_syndrs_struct[cell_ix][2][10] = primal_qbts[29]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][11] = self.cells_qbts_struct[neigh_cell][27]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][12] = self.cells_qbts_struct[neigh_cell][35]

            self.cells_primal_syndrs_struct[cell_ix][2][13] = primal_qbts[33]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][14] = self.cells_qbts_struct[neigh_cell][34]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][15] = self.cells_qbts_struct[neigh_cell][0]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][16] = self.cells_qbts_struct[neigh_cell][2]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][17] = self.cells_qbts_struct[neigh_cell][1]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][18] = self.cells_qbts_struct[neigh_cell][4]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][19] = self.cells_qbts_struct[neigh_cell][5]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][20] = self.cells_qbts_struct[neigh_cell][3]

            if self.chains_len == 4:
                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
                neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x')
                self.cells_primal_syndrs_struct[cell_ix][2][21] = self.cells_qbts_struct[neigh_cell][7]

                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
                self.cells_primal_syndrs_struct[cell_ix][2][22] = self.cells_qbts_struct[neigh_cell][8]

                neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
                neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y')
                self.cells_primal_syndrs_struct[cell_ix][2][23] = self.cells_qbts_struct[neigh_cell][6]

    # Function to calculate the logical operators of the lattice
    def get_logical_operators(self):
        log_x = []
        log_y = []
        log_z = []

        # build logical operator surface on z direction
        z_ix = 0
        for y_ix in range(self.lattice_y_size):
            for x_ix in range(self.lattice_x_size):
                cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                primal_qubits = self.cells_qbts_struct[cell_ix]
                if self.chains_len == 4:
                    log_z += list(primal_qubits[[0, 1, 2, 3, 4, 5, 6, 7, 8]])
                else:
                    log_z += list(primal_qubits[[0, 1, 2, 3, 4, 5]])

                shifted_cell_ix = self.shifted_cell_ix(cell_ix, -1, 'z')
                primal_qubits = self.cells_qbts_struct[shifted_cell_ix]
                if self.chains_len == 4:
                    log_z += list(primal_qubits[[30, 31, 32, 33, 34, 35]])
                else:
                    log_z += list(primal_qubits[[33, 34, 35]])

        self.log_ops_qbts = [log_x, log_y, log_z]

    # Functions that build the matching matrix for syndromes of the lattice
    def get_matching_matrix(self):
        h_matrix = np.zeros((3 * self.num_cells, self.num_primal_qbts), dtype=np.uint8)
        for cell_ix, cell_syndrs in enumerate(self.cells_primal_syndrs_struct):
            for syndr_ix, syndr_nodes_list in enumerate(cell_syndrs):
                syndr_qubits = syndr_nodes_list[syndr_nodes_list != self.empty_ix]
                h_matrix[3 * cell_ix + syndr_ix, syndr_qubits] = np.ones(len(syndr_qubits), dtype=np.uint8)
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
        est_num_qbts_log = np.log2(18 * self.num_cells)
        if est_num_qbts_log < 8:
            data_type = np.uint8
        elif est_num_qbts_log < 16:
            data_type = np.uint16
        elif est_num_qbts_log < 32:
            data_type = np.uint32
        else:
            data_type = np.uint64
        return data_type, np.iinfo(data_type).max