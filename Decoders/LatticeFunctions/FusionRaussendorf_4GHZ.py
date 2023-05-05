import numpy as np

class FusionLattice_Raussendorf_4GHZ(object):

    def __init__(self, lattice_x_size, lattice_y_size, lattice_t_size):
        # boundary is considered to be periodic, just builds primal Fusion lattice for simplicity

        self.lattice_x_size = lattice_x_size
        self.lattice_y_size = lattice_y_size
        self.lattice_t_size = lattice_t_size

        self.num_primal_cells = lattice_x_size * lattice_y_size * lattice_t_size
        self.data_type, self.empty_ix = self.get_data_type()

        # 3d spatial elementary shifts to move between lattice nodes
        self.xshift = np.array((0.5, 0, 0))
        self.yshift = np.array((0, 0.5, 0))
        self.tshift = np.array((0, 0, 0.5))

        self.primal_nodes_positions = []
        self.edged_qubits = []
        self.cells_shapes = []

        self.num_primal_fusions = 0
        self.tot_num_fusions = 0

        self.current_primal_fus_ix = 0

        self.primal_cells_fus_struct = np.array([[self.empty_ix] * 6 for _ in range(self.num_primal_cells)],
                                                 dtype=self.data_type)

        self.log_ops_fus = [None, None]

        self.build_lattice_fusions()
        self.get_logical_operators()


    def build_lattice_fusions(self):
        # Starts building the lattice
        for t_ix in range(self.lattice_t_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):

                    # adding primal fusions
                    this_cell_pos = np.array((x_ix + 0.5, y_ix + 0.5, t_ix + 0.5))
                    this_cell_ix = self.cell_xytcoords_to_ix([x_ix, y_ix, t_ix])

                    ## x direction
                    this_node_pos = this_cell_pos - self.xshift
                    self.primal_nodes_positions.append(this_node_pos)
                    self.primal_cells_fus_struct[this_cell_ix][0] = self.current_primal_fus_ix
                    neigh_cell = self.shifted_cell_ix(this_cell_ix, -1, 'x')
                    self.primal_cells_fus_struct[neigh_cell][1] = self.current_primal_fus_ix
                    self.current_primal_fus_ix += 1


                    ## y direction
                    this_node_pos = this_cell_pos - self.yshift
                    self.primal_nodes_positions.append(this_node_pos)
                    self.primal_cells_fus_struct[this_cell_ix][2] = self.current_primal_fus_ix
                    neigh_cell = self.shifted_cell_ix(this_cell_ix, -1, 'y')
                    self.primal_cells_fus_struct[neigh_cell][3] = self.current_primal_fus_ix
                    self.current_primal_fus_ix += 1

                    ## z direction
                    this_node_pos = this_cell_pos - self.tshift
                    self.primal_nodes_positions.append(this_node_pos)
                    self.primal_cells_fus_struct[this_cell_ix][4] = self.current_primal_fus_ix
                    neigh_cell = self.shifted_cell_ix(this_cell_ix, -1, 't')
                    self.primal_cells_fus_struct[neigh_cell][5] = self.current_primal_fus_ix
                    self.current_primal_fus_ix += 1

                    # adding primal cell shapes
                    self.cells_shapes.append(
                        np.array((this_cell_pos - self.xshift - self.yshift - self.tshift,
                                  this_cell_pos + self.xshift - self.yshift - self.tshift)))
                    self.cells_shapes.append(
                        np.array((this_cell_pos - self.xshift - self.yshift - self.tshift,
                                  this_cell_pos - self.xshift + self.yshift - self.tshift)))
                    self.cells_shapes.append(
                        np.array((this_cell_pos - self.xshift - self.yshift - self.tshift,
                                  this_cell_pos - self.xshift - self.yshift + self.tshift)))

                    if t_ix == (self.lattice_t_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos - self.xshift - self.yshift + self.tshift,
                             this_cell_pos + self.xshift - self.yshift + self.tshift)))
                    if y_ix == (self.lattice_y_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos - self.xshift + self.yshift - self.tshift,
                             this_cell_pos + self.xshift + self.yshift - self.tshift)))
                    if y_ix == (self.lattice_y_size - 1) and t_ix == (self.lattice_t_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos - self.xshift + self.yshift + self.tshift,
                             this_cell_pos + self.xshift + self.yshift + self.tshift)))

                    if t_ix == (self.lattice_t_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos - self.xshift - self.yshift + self.tshift,
                             this_cell_pos - self.xshift + self.yshift + self.tshift)))
                    if x_ix == (self.lattice_x_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos + self.xshift - self.yshift - self.tshift,
                             this_cell_pos + self.xshift + self.yshift - self.tshift)))
                    if x_ix == (self.lattice_x_size - 1) and t_ix == (self.lattice_t_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos + self.xshift - self.yshift + self.tshift,
                             this_cell_pos + self.xshift + self.yshift + self.tshift)))

                    if y_ix == (self.lattice_y_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos - self.xshift + self.yshift - self.tshift,
                             this_cell_pos - self.xshift + self.yshift + self.tshift)))
                    if x_ix == (self.lattice_x_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos + self.xshift - self.yshift - self.tshift,
                             this_cell_pos + self.xshift - self.yshift + self.tshift)))
                    if x_ix == (self.lattice_x_size - 1) and y_ix == (self.lattice_y_size - 1):
                        self.cells_shapes.append(np.array(
                            (this_cell_pos + self.xshift + self.yshift - self.tshift,
                             this_cell_pos + self.xshift + self.yshift + self.tshift)))

        self.num_primal_fusions = self.current_primal_fus_ix
        self.tot_num_fusions = self.num_primal_fusions

        self.primal_nodes_positions = np.array(self.primal_nodes_positions)


    def get_logical_operators(self):
        log_x = None
        log_z = None
        # build logical operator surface on x direction

        fus_pos = self.primal_nodes_positions
        target_x_pos = 0.
        log_x = np.where(np.array(list(map(lambda x: x[0] == target_x_pos, fus_pos))))[0]
        self.log_ops_fus = [log_x, log_z]

    # Functions that build the matching matrix for syndromes of the lattice

    def get_matching_matrix(self):
        num_cells = self.num_primal_cells
        num_fusions = self.num_primal_fusions
        cells_struct = self.primal_cells_fus_struct
        H_matrix = np.zeros((num_cells, num_fusions), dtype=np.uint8)
        for syndr_ix, syndr_nodes_list in enumerate(cells_struct):
            syndr_fusions = syndr_nodes_list[syndr_nodes_list != self.empty_ix]
            H_matrix[syndr_ix, syndr_fusions] = np.ones(len(syndr_fusions), dtype=np.uint8)
        return H_matrix

    # Functions to deal with cells in Raussendorf lattice

    def cell_xytcoords_to_ix(self, cell_xytcoords):
        return cell_xytcoords[0] + cell_xytcoords[1] * self.lattice_x_size \
               + cell_xytcoords[2] * self.lattice_x_size * self.lattice_y_size

    def cell_ix_to_xytcoords(self, cell_ix):
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
            size_lim = self.lattice_x_size - 1
        elif shift_axis == 'y':
            axis_label = 1
            size_lim = self.lattice_y_size - 1
        elif shift_axis == 't':
            axis_label = 2
            size_lim = self.lattice_t_size - 1
        else:
            raise ValueError('Shift axis can only be one of (x, y, or t)')

        temp_coords = self.cell_ix_to_xytcoords(cell_ix)
        # boundary is considered periodic
        temp_coords[axis_label] = (temp_coords[axis_label] + shift) % (size_lim+1)
        return self.cell_xytcoords_to_ix(temp_coords)

    # Data type handler
    def get_data_type(self):
        est_num_fus_log = np.log2(3 * self.num_primal_cells)
        if est_num_fus_log < 8:
            data_type = np.uint8
        elif est_num_fus_log < 16:
            data_type = np.uint16
        elif est_num_fus_log < 32:
            data_type = np.uint32
        else:
            data_type = np.uint64
        return data_type, np.iinfo(data_type).max
