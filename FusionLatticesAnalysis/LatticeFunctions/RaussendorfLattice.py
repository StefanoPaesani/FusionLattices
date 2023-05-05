import numpy as np
import warnings

import matplotlib.pyplot as plt


################################################
########## Raussendorf Lattice Class  ##########
################################################


class RaussendorfLattice(object):
    r"""
    Class building and representing a Raussendorf lattice

    As inputs it takes the lattice sizes, i.e. number of primal cells in the x, y and z directions,
    and the type of the boundries.

    'num_primal_cells': total number of primal cells
    'num_primal_qbts': total number of primal physical qubits
    'num_dual_cells': total number of dual cells
    'num_dual_qbts': total number of dual physical qubits
    'tot_num_qbts': total of number of qubits, including primal and dual

    'data_type': type of data used to label the qubits in the lattice.
    'empty_ix': label in data_type used to indicate empty qubit slots in the lattice

    'primal_nodes_positions': positions (in 3d space) of the primal qubits
    'dual_nodes_positions': positions (in 3d space) of the dual qubits

    'log_ops_qbts': contains a list of primal qubits forming a X logical operators, and a list of duals forming Z.

    'primal_cells_qbts_struct': Structure that contains the primal qubits contained in each primal cell of the lattice,
    ordered according to their position in the lattice.
    'dual_cells_qbts_struct': Structure that contains the dual qubits contained in each dual cell of the lattice,
    ordered according to their position in the lattice.
    #### Order of qubits in cells struct is [[x-, x+, y-, y+, t-, t+] (cell0), [x-, x+, y-, y+, t-, t+] (cell1), ... ]
    #### Empty qubits are initialized in the max data_type number empty_ix (e.g. 65535 if np.int16 is used)

    'edged_qubits': list containing all pairs of primal and dual qubits connected by an edge, in the form of
    [[primal_qbt0, dual_qbt0], [primal_qbt1, dual_qbt1], ... ]

    'cell_shapes': shapes (edges) of the primal lattice cells used when plotting the lattice
    'plotted_lattice_edges': lattice edges used when plotting the lattice

    """

    def __init__(self, lattice_x_size, lattice_y_size, lattice_t_size,
                 boundary_is_primal=((True, True), (True, True), (True, True)), boundary_is_periodic=False):
        # boundary_is_primal indicates the type for each boundry of the lattice,
        # ordered as [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

        self.lattice_x_size = lattice_x_size
        self.lattice_y_size = lattice_y_size
        self.lattice_t_size = lattice_t_size

        self.boundary_is_periodic = boundary_is_periodic
        if self.boundary_is_periodic:
            if boundary_is_primal != ((True, False), (True, False), (True, False)):
                warnings.warn("The boundaries were set to periodic, but boundary_is_primal not to "
                              "((True, False), (True, False), (True, False)). Setting it to these values.")
            self.boundary_is_primal = ((True, False), (True, False), (True, False))
        else:
            self.boundary_is_primal = boundary_is_primal

        self.num_primal_cells = lattice_x_size * lattice_y_size * lattice_t_size
        self.data_type, self.empty_ix = self.get_data_type()

        # The center of the 1st dual lattice cell is assumed to be on the x-, y-, z- position of the first primal cell.
        self.dual_lattice_x_size = lattice_x_size + (1 if boundary_is_primal[0][1] else 0)
        self.dual_lattice_y_size = lattice_y_size + (1 if boundary_is_primal[1][1] else 0)
        self.dual_lattice_t_size = lattice_t_size + (1 if boundary_is_primal[2][1] else 0)

        self.num_dual_cells = self.dual_lattice_x_size * self.dual_lattice_y_size * self.dual_lattice_t_size

        # 3d spatial elementary shifts to move between lattice nodes
        self.xshift = np.array((0.5, 0, 0))
        self.yshift = np.array((0, 0.5, 0))
        self.tshift = np.array((0, 0, 0.5))

        self.primal_nodes_positions = []
        self.dual_nodes_positions = []
        self.edged_qubits = []
        self.plotted_lattice_edges = []
        self.cells_shapes = []

        self.primal_boundary_qbts = [[[], []], [[], []], [[], []]]
        # TODO: dual_boundary_qbts not implemented yet when building the Raussendorf lattice
        self.dual_boundary_qbts = [[[], []], [[], []], [[], []]]

        self.num_primal_qbts = 0
        self.num_dual_qbts = 0
        self.tot_num_qbts = 0

        self.current_primal_qbt_ix = 0
        self.current_dual_qbt_ix = 0

        self.primal_cells_qbts_struct = np.array([[self.empty_ix] * 6 for _ in range(self.num_primal_cells)],
                                                 dtype=self.data_type)
        self.dual_cells_qbts_struct = np.array([[self.empty_ix] * 6 for _ in range(self.num_dual_cells)],
                                               dtype=self.data_type)

        self.log_ops_qbts = [None, None]

        self.build_lattice_nodes()
        self.build_lattice_edges()
        self.get_logical_operators()

    def build_lattice_nodes(self):
        # Starts building the lattice
        for t_ix in range(self.lattice_t_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):

                    # adding primal nodes
                    this_cell_pos = np.array((x_ix + 0.5, y_ix + 0.5, t_ix + 0.5))
                    this_cell_ix = self.cell_xytcoords_to_ix([x_ix, y_ix, t_ix])

                    if x_ix > 0 or self.boundary_is_primal[0][0]:
                        this_node_pos = this_cell_pos - self.xshift
                        self.primal_nodes_positions.append(this_node_pos)
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.yshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.yshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.tshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.tshift)))
                        self.primal_cells_qbts_struct[this_cell_ix][0] = self.current_primal_qbt_ix
                        if x_ix == 0 and self.boundary_is_primal[0][0]:
                            self.primal_boundary_qbts[0][0].append(self.current_primal_qbt_ix)
                        try:
                            neigh_cell = self.shifted_cell_ix(this_cell_ix, -1, 'x')
                            self.primal_cells_qbts_struct[neigh_cell][1] = self.current_primal_qbt_ix
                        except IndexError:
                            pass
                        self.current_primal_qbt_ix += 1
                    if y_ix > 0 or self.boundary_is_primal[1][0]:
                        this_node_pos = this_cell_pos - self.yshift
                        self.primal_nodes_positions.append(this_node_pos)
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.tshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.tshift)))
                        self.primal_cells_qbts_struct[this_cell_ix][2] = self.current_primal_qbt_ix
                        if y_ix == 0 and self.boundary_is_primal[1][0]:
                            self.primal_boundary_qbts[1][0].append(self.current_primal_qbt_ix)
                        try:
                            neigh_cell = self.shifted_cell_ix(this_cell_ix, -1, 'y')
                            self.primal_cells_qbts_struct[neigh_cell][3] = self.current_primal_qbt_ix
                        except IndexError:
                            pass
                        self.current_primal_qbt_ix += 1
                    if t_ix > 0 or self.boundary_is_primal[2][0]:
                        this_node_pos = this_cell_pos - self.tshift
                        self.primal_nodes_positions.append(this_node_pos)
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.yshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.yshift)))
                        self.primal_cells_qbts_struct[this_cell_ix][4] = self.current_primal_qbt_ix
                        if t_ix == 0 and self.boundary_is_primal[2][0]:
                            self.primal_boundary_qbts[2][0].append(self.current_primal_qbt_ix)
                        try:
                            neigh_cell = self.shifted_cell_ix(this_cell_ix, -1, 't')
                            self.primal_cells_qbts_struct[neigh_cell][5] = self.current_primal_qbt_ix
                        except IndexError:
                            pass
                        self.current_primal_qbt_ix += 1

                    if x_ix == (self.lattice_x_size - 1) and self.boundary_is_primal[0][1]:
                        this_node_pos = this_cell_pos + self.xshift
                        self.primal_nodes_positions.append(this_node_pos)
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.yshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.yshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.tshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.tshift)))
                        self.primal_cells_qbts_struct[this_cell_ix][1] = self.current_primal_qbt_ix
                        self.primal_boundary_qbts[0][1].append(self.current_primal_qbt_ix)
                        self.current_primal_qbt_ix += 1
                    if y_ix == (self.lattice_y_size - 1) and self.boundary_is_primal[1][1]:
                        this_node_pos = this_cell_pos + self.yshift
                        self.primal_nodes_positions.append(this_node_pos)
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.tshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.tshift)))
                        self.primal_cells_qbts_struct[this_cell_ix][3] = self.current_primal_qbt_ix
                        self.primal_boundary_qbts[1][1].append(self.current_primal_qbt_ix)
                        self.current_primal_qbt_ix += 1
                    if t_ix == (self.lattice_t_size - 1) and self.boundary_is_primal[2][1]:
                        this_node_pos = this_cell_pos + self.tshift
                        self.primal_nodes_positions.append(this_node_pos)
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.xshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos - self.yshift)))
                        self.plotted_lattice_edges.append(np.array((this_node_pos, this_node_pos + self.yshift)))
                        self.primal_cells_qbts_struct[this_cell_ix][5] = self.current_primal_qbt_ix
                        self.primal_boundary_qbts[2][1].append(self.current_primal_qbt_ix)
                        self.current_primal_qbt_ix += 1

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

                    # adding dual nodes
                    this_dual_cell_pos = np.array((x_ix, y_ix, t_ix))
                    this_dual_cell_ix = self.cell_xytcoords_to_ix([x_ix, y_ix, t_ix], 'dual')

                    if (x_ix > 0 or self.boundary_is_primal[0][0]) and (y_ix > 0 or self.boundary_is_primal[1][0]):
                        this_dual_node_pos = this_dual_cell_pos + self.tshift
                        self.dual_nodes_positions.append(this_dual_node_pos)
                        self.dual_cells_qbts_struct[this_dual_cell_ix][5] = self.current_dual_qbt_ix
                        try:
                            neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 't', 'dual')
                            self.dual_cells_qbts_struct[neigh_cell][4] = self.current_dual_qbt_ix
                        except IndexError:
                            pass
                        self.current_dual_qbt_ix += 1

                    if (x_ix > 0 or self.boundary_is_primal[0][0]) and (t_ix > 0 or self.boundary_is_primal[2][0]):
                        this_dual_node_pos = this_dual_cell_pos + self.yshift
                        self.dual_nodes_positions.append(this_dual_node_pos)
                        self.dual_cells_qbts_struct[this_dual_cell_ix][3] = self.current_dual_qbt_ix
                        try:
                            neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 'y', 'dual')
                            self.dual_cells_qbts_struct[neigh_cell][2] = self.current_dual_qbt_ix
                        except IndexError:
                            pass
                        self.current_dual_qbt_ix += 1

                    if (y_ix > 0 or self.boundary_is_primal[1][0]) and (t_ix > 0 or self.boundary_is_primal[2][0]):
                        this_dual_node_pos = this_dual_cell_pos + self.xshift
                        self.dual_nodes_positions.append(this_dual_node_pos)
                        self.dual_cells_qbts_struct[this_dual_cell_ix][1] = self.current_dual_qbt_ix
                        try:
                            neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 'x', 'dual')
                            self.dual_cells_qbts_struct[neigh_cell][0] = self.current_dual_qbt_ix
                        except IndexError:
                            pass
                        self.current_dual_qbt_ix += 1

                    if x_ix == (self.lattice_x_size - 1) and self.boundary_is_primal[0][1]:
                        neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 'x', 'dual')
                        neigh_cell_pos = np.array((x_ix + 1, y_ix, t_ix))

                        if t_ix > 0 or self.boundary_is_primal[2][0]:
                            this_dual_node_pos = neigh_cell_pos + self.yshift
                            self.dual_nodes_positions.append(this_dual_node_pos)
                            self.dual_cells_qbts_struct[neigh_cell][3] = self.current_dual_qbt_ix
                            try:
                                new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y', 'dual')
                                self.dual_cells_qbts_struct[new_neigh_cell][2] = self.current_dual_qbt_ix
                            except IndexError:
                                pass
                            self.current_dual_qbt_ix += 1

                        if y_ix > 0 or self.boundary_is_primal[1][0]:
                            this_dual_node_pos = neigh_cell_pos + self.tshift
                            self.dual_nodes_positions.append(this_dual_node_pos)
                            self.dual_cells_qbts_struct[neigh_cell][5] = self.current_dual_qbt_ix
                            try:
                                new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 't', 'dual')
                                self.dual_cells_qbts_struct[new_neigh_cell][4] = self.current_dual_qbt_ix
                            except IndexError:
                                pass
                            self.current_dual_qbt_ix += 1

                    if y_ix == (self.lattice_y_size - 1) and self.boundary_is_primal[1][1]:
                        neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 'y', 'dual')
                        neigh_cell_pos = np.array((x_ix, y_ix + 1, t_ix))

                        if t_ix > 0 or self.boundary_is_primal[2][0]:

                            this_dual_node_pos = neigh_cell_pos + self.xshift
                            self.dual_nodes_positions.append(this_dual_node_pos)
                            self.dual_cells_qbts_struct[neigh_cell][1] = self.current_dual_qbt_ix
                            try:
                                new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x', 'dual')
                                self.dual_cells_qbts_struct[new_neigh_cell][0] = self.current_dual_qbt_ix
                            except IndexError:
                                pass
                            self.current_dual_qbt_ix += 1

                        if x_ix > 0 or self.boundary_is_primal[0][0]:
                            this_dual_node_pos = neigh_cell_pos + self.tshift
                            self.dual_nodes_positions.append(this_dual_node_pos)
                            self.dual_cells_qbts_struct[neigh_cell][5] = self.current_dual_qbt_ix
                            try:
                                new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 't', 'dual')
                                self.dual_cells_qbts_struct[new_neigh_cell][4] = self.current_dual_qbt_ix
                            except IndexError:
                                pass
                            self.current_dual_qbt_ix += 1

                    if t_ix == (self.lattice_t_size - 1) and self.boundary_is_primal[2][1]:
                        neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 't', 'dual')
                        neigh_cell_pos = np.array((x_ix, y_ix, t_ix + 1))

                        if y_ix > 0 or self.boundary_is_primal[1][0]:
                            this_dual_node_pos = neigh_cell_pos + self.xshift
                            self.dual_nodes_positions.append(this_dual_node_pos)
                            self.dual_cells_qbts_struct[neigh_cell][1] = self.current_dual_qbt_ix
                            try:
                                new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x', 'dual')
                                self.dual_cells_qbts_struct[new_neigh_cell][0] = self.current_dual_qbt_ix
                            except IndexError:
                                pass
                            self.current_dual_qbt_ix += 1

                        if x_ix > 0 or self.boundary_is_primal[0][0]:
                            this_dual_node_pos = neigh_cell_pos + self.yshift
                            self.dual_nodes_positions.append(this_dual_node_pos)
                            self.dual_cells_qbts_struct[neigh_cell][3] = self.current_dual_qbt_ix
                            try:
                                new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y', 'dual')
                                self.dual_cells_qbts_struct[new_neigh_cell][2] = self.current_dual_qbt_ix
                            except IndexError:
                                pass
                            self.current_dual_qbt_ix += 1

                    if (x_ix == (self.lattice_x_size - 1) and self.boundary_is_primal[0][1]) and (
                            y_ix == (self.lattice_y_size - 1) and self.boundary_is_primal[1][1]):
                        neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 'x', 'dual')
                        neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y', 'dual')
                        neigh_cell_pos = np.array((x_ix + 1, y_ix + 1, t_ix))

                        this_dual_node_pos = neigh_cell_pos + self.tshift
                        self.dual_nodes_positions.append(this_dual_node_pos)
                        self.dual_cells_qbts_struct[neigh_cell][5] = self.current_dual_qbt_ix
                        try:
                            new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 't', 'dual')
                            self.dual_cells_qbts_struct[new_neigh_cell][4] = self.current_dual_qbt_ix
                        except IndexError:
                            pass
                        self.current_dual_qbt_ix += 1

                    if (x_ix == (self.lattice_x_size - 1) and self.boundary_is_primal[0][1]) and (
                            t_ix == (self.lattice_t_size - 1) and self.boundary_is_primal[2][1]):
                        neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 'x', 'dual')
                        neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 't', 'dual')
                        neigh_cell_pos = np.array((x_ix + 1, y_ix, t_ix + 1))

                        this_dual_node_pos = neigh_cell_pos + self.yshift
                        self.dual_nodes_positions.append(this_dual_node_pos)
                        self.dual_cells_qbts_struct[neigh_cell][3] = self.current_dual_qbt_ix
                        try:
                            new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'y', 'dual')
                            self.dual_cells_qbts_struct[new_neigh_cell][2] = self.current_dual_qbt_ix
                        except IndexError:
                            pass
                        self.current_dual_qbt_ix += 1

                    if (y_ix == (self.lattice_y_size - 1) and self.boundary_is_primal[1][1]) and (
                            t_ix == (self.lattice_t_size - 1) and self.boundary_is_primal[2][1]):
                        neigh_cell = self.shifted_cell_ix(this_dual_cell_ix, +1, 'y', 'dual')
                        neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 't', 'dual')
                        neigh_cell_pos = np.array((x_ix, y_ix + 1, t_ix + 1))

                        this_dual_node_pos = neigh_cell_pos + self.xshift
                        self.dual_nodes_positions.append(this_dual_node_pos)
                        self.dual_cells_qbts_struct[neigh_cell][1] = self.current_dual_qbt_ix
                        try:
                            new_neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'x', 'dual')
                            self.dual_cells_qbts_struct[new_neigh_cell][0] = self.current_dual_qbt_ix
                        except IndexError:
                            pass
                        self.current_dual_qbt_ix += 1

        self.num_primal_qbts = self.current_primal_qbt_ix
        self.num_dual_qbts = self.current_dual_qbt_ix
        # noinspection PyInterpreter
        self.tot_num_qbts = self.num_primal_qbts + self.num_dual_qbts

        self.primal_nodes_positions = np.array(self.primal_nodes_positions)
        self.dual_nodes_positions = np.array(self.dual_nodes_positions)

    # Function that builds list of connected qubits
    # ordered as [[prima_qubit_ix1, dual_qubit_ix1], [prima_qubit_ix2, dual_qubit_ix2], ...]

    def build_lattice_edges(self, primal_to_dual_edges_policy_dict=None):
        if primal_to_dual_edges_policy_dict is None:
            # This dict contains, for all primal qubits in a cell the dual qbt type label and the associated cell shift
            # for all possible 4 dual qubits it could be connected to.
            # Primal qubits type labels are the keys and the values are of the type
            # ((dual_qbt_type_label1, prima_to_dual_cell_shift1),...,(dual_qbt_type_label4, prima_to_dual_cell_shift4))
            primal_to_dual_edges_policy_dict = {
                0: ((5, (0, 0, 0)), (3, (0, 0, 0)), (5, (0, 1, 0)), (3, (0, 0, 1))),
                1: ((5, (1, 0, 0)), (3, (1, 0, 0)), (5, (1, 1, 0)), (3, (1, 0, 1))),
                2: ((5, (0, 0, 0)), (1, (0, 0, 0)), (5, (1, 0, 0)), (1, (0, 0, 1))),
                3: ((5, (0, 1, 0)), (1, (0, 1, 0)), (5, (1, 1, 0)), (1, (0, 1, 1))),
                4: ((3, (0, 0, 0)), (1, (0, 0, 0)), (3, (1, 0, 0)), (1, (0, 1, 0))),
                5: ((3, (0, 0, 1)), (1, (0, 0, 1)), (3, (1, 0, 1)), (1, (0, 1, 1)))
            }

        for t_ix in range(self.lattice_t_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):
                    primal_cell_ix = self.cell_xytcoords_to_ix([x_ix, y_ix, t_ix])
                    temp_used_primal_labels = []

                    if x_ix > 0 or self.boundary_is_primal[0][0]:
                        temp_used_primal_labels.append(0)
                    if y_ix > 0 or self.boundary_is_primal[1][0]:
                        temp_used_primal_labels.append(2)
                    if t_ix > 0 or self.boundary_is_primal[2][0]:
                        temp_used_primal_labels.append(4)

                    if x_ix == (self.lattice_x_size - 1) and self.boundary_is_primal[0][1]:
                        temp_used_primal_labels.append(1)
                    if y_ix == (self.lattice_y_size - 1) and self.boundary_is_primal[1][1]:
                        temp_used_primal_labels.append(3)
                    if t_ix == (self.lattice_t_size - 1) and self.boundary_is_primal[2][1]:
                        temp_used_primal_labels.append(5)

                    temp_edged_qbts = self.find_all_qbts_edges_in_primal_cell(
                        primal_cell_ix, temp_used_primal_labels, primal_to_dual_edges_policy_dict)

                    self.edged_qubits = self.edged_qubits + temp_edged_qbts

    # Function to calculate the logical operators of the lattice

    def get_logical_operators(self):
        log_x = None
        log_z = None
        # build logical operator surface on x direction
        if self.boundary_is_primal[0][0] or self.boundary_is_primal[0][1]:
            qbts_pos = self.primal_nodes_positions
            target_x_pos = 0.
            log_x = np.where(np.array(list(map(lambda x: x[0] == target_x_pos, qbts_pos))))[0]
        elif (not self.boundary_is_primal[0][0]) and (not self.boundary_is_primal[0][1]):
            qbts_pos = self.dual_nodes_positions
            target_x_pos = 0.5
            log_z = np.where(np.array(list(map(lambda x: x[0] == target_x_pos, qbts_pos))))[0]

        # build logical operator surface on y direction
        if self.boundary_is_primal[1][0] and self.boundary_is_primal[1][1]:
            qbts_pos = self.primal_nodes_positions
            target_y_pos = 0.
            log_x = np.where(np.array(list(map(lambda y: y[1] == target_y_pos, qbts_pos))))[0]
        elif (not self.boundary_is_primal[1][0]) or (not self.boundary_is_primal[1][1]):
            qbts_pos = self.dual_nodes_positions
            target_y_pos = 0.5
            log_z = np.where(np.array(list(map(lambda y: y[1] == target_y_pos, qbts_pos))))[0]
        self.log_ops_qbts = [log_x, log_z]


    # Functions to deal with cells in Raussendorf lattice

    def cell_xytcoords_to_ix(self, cell_xytcoords, latt_type='primal'):
        if latt_type == 'primal':
            size_x = self.lattice_x_size
            size_y = self.lattice_y_size
        elif latt_type == 'dual':
            size_x = self.dual_lattice_x_size
            size_y = self.dual_lattice_y_size
        else:
            raise ValueError('latt_type can be only "primal" or "dual"')

        return cell_xytcoords[0] + cell_xytcoords[1] * size_x + cell_xytcoords[2] * size_x * size_y

    def cell_ix_to_xytcoords(self, cell_ix, latt_type='primal'):
        if latt_type == 'primal':
            size_x = self.lattice_x_size
            size_y = self.lattice_y_size
        elif latt_type == 'dual':
            size_x = self.dual_lattice_x_size
            size_y = self.dual_lattice_y_size
        else:
            raise ValueError('latt_type can be only "primal" or "dual"')

        t_coord = int(cell_ix / (size_x * size_y))
        y_coord = int(cell_ix / size_x) % size_y
        x_coord = cell_ix % size_x
        return np.array((x_coord, y_coord, t_coord))

    def shifted_cell_ix(self, cell_ix, shift, shift_axis, latt_type='primal'):
        #     Function to get the index of a cell obtained shifting an initial cell with 'cell_ix'
        #     by a integer (positive or negative) step 'shift' along an axis 'shift_axis' in x, y, or t.
        if not isinstance(shift, int):
            raise ValueError('The parameter shift can only be an integer (positive or negative)')

        if latt_type == 'primal':
            size_x = self.lattice_x_size
            size_y = self.lattice_y_size
            size_t = self.lattice_t_size
        elif latt_type == 'dual':
            size_x = self.dual_lattice_x_size
            size_y = self.dual_lattice_y_size
            size_t = self.dual_lattice_t_size
        else:
            raise ValueError('latt_type can be only "primal" or "dual"')

        if shift_axis == 'x':
            axis_label = 0
            size_lim = size_x - 1
        elif shift_axis == 'y':
            axis_label = 1
            size_lim = size_y - 1
        elif shift_axis == 't':
            axis_label = 2
            size_lim = size_t - 1
        else:
            raise ValueError('Shift axis can only be one of (x, y, or t)')

        temp_coords = self.cell_ix_to_xytcoords(cell_ix, latt_type)
        # print('old temp_coords', temp_coords)
        if self.boundary_is_periodic:
            # print(temp_coords[axis_label], shift, size_lim+1, temp_coords[axis_label] + shift, (temp_coords[axis_label] + shift) % (size_lim+1))
            temp_coords[axis_label] = (temp_coords[axis_label] + shift) % (size_lim+1)
            # print('new temp_coords', temp_coords)
            return self.cell_xytcoords_to_ix(temp_coords, latt_type)
        else:
            if 0 <= (temp_coords[axis_label] + shift) <= size_lim:
                temp_coords[axis_label] = temp_coords[axis_label] + shift
                return self.cell_xytcoords_to_ix(temp_coords, latt_type)
            else:
                raise IndexError('Shifted cell is out of boundries of lattice size (%d, %d, %d)' % (size_x, size_y, size_t))

    # Functions to find connected qubits in Raussendorf lattice

    def try_find_qbts_for_edge(self, primal_cell_ix, primal_to_dual_cell_shift, primal_cell_qbt_type_label,
                               dual_cell_qbt_type_label):
        primal_cell_coords = self.cell_ix_to_xytcoords(primal_cell_ix)

        dual_cell_coords = np.array(primal_cell_coords) + np.array(primal_to_dual_cell_shift)
        if np.all(np.logical_and(dual_cell_coords <= (np.array(
                (self.dual_lattice_x_size, self.dual_lattice_y_size, self.dual_lattice_t_size),
                dtype=self.data_type) - 1),
                                 dual_cell_coords >= 0)):
            dual_cell_ix = self.cell_xytcoords_to_ix(dual_cell_coords)

            primal_qbt_ix = self.primal_cells_qbts_struct[primal_cell_ix][primal_cell_qbt_type_label]
            dual_qbt_ix = self.dual_cells_qbts_struct[dual_cell_ix][dual_cell_qbt_type_label]
            if primal_qbt_ix != self.empty_ix and dual_qbt_ix != self.empty_ix:
                return True, [primal_qbt_ix, dual_qbt_ix]
            else:
                return False, None
        else:
            return False, None

    def find_all_qbts_edges_in_primal_cell(self, primal_cell_ix, used_primal_qbt_type_labels,
                                           primal_to_dual_edges_policy_dict=None):

        if primal_to_dual_edges_policy_dict is None:
            primal_to_dual_edges_policy_dict = {
                0: ((5, (0, 0, 0)), (3, (0, 0, 0)), (5, (0, 1, 0)), (3, (0, 0, 1))),
                1: ((5, (1, 0, 0)), (3, (1, 0, 0)), (5, (1, 1, 0)), (3, (1, 0, 1))),
                2: ((5, (0, 0, 0)), (1, (0, 0, 0)), (5, (1, 0, 0)), (1, (0, 0, 1))),
                3: ((5, (0, 1, 0)), (1, (0, 1, 0)), (5, (1, 1, 0)), (1, (0, 1, 1))),
                4: ((3, (0, 0, 0)), (1, (0, 0, 0)), (3, (1, 0, 0)), (1, (0, 1, 0))),
                5: ((3, (0, 0, 1)), (1, (0, 0, 1)), (3, (1, 0, 1)), (1, (0, 1, 1)))
            }

        found_edges = []
        for primal_cell_qbt_type_label in used_primal_qbt_type_labels:
            for [dual_cell_qbt_type_label, prima_to_dual_cell_shift] in primal_to_dual_edges_policy_dict[
                                                                                        primal_cell_qbt_type_label]:
                edge_exists, edge_qbts = self.try_find_qbts_for_edge(primal_cell_ix, prima_to_dual_cell_shift,
                                                                     primal_cell_qbt_type_label,
                                                                     dual_cell_qbt_type_label)
                if edge_exists:
                    found_edges.append(edge_qbts)
        return found_edges

    # Data type handler

    def get_data_type(self):
        est_num_qbts_log = np.log2(6 * self.num_primal_cells)
        if est_num_qbts_log < 8:
            data_type = np.uint8
        elif est_num_qbts_log < 16:
            data_type = np.uint16
        elif est_num_qbts_log < 32:
            data_type = np.uint32
        else:
            data_type = np.uint64
        return data_type, np.iinfo(data_type).max

    # Function to draw the Raussendorf lattice

    def draw_lattice(self, print_nodes=True, print_edges=True, print_cells=True, fig_size=10, fig_aspect=None):
        if fig_aspect is None:
            fig_aspect = self.lattice_t_size / ((self.lattice_x_size + self.lattice_y_size) / 2)

        node_size = 200 / (self.lattice_x_size * self.lattice_y_size)
        primal_node_color = 'black'  # 'red'
        dual_node_color = 'blue'

        edge_color = 'black'
        edge_alpha = 1
        edge_width = 1

        cell_color = 'black'
        cell_alpha = 0.5
        cell_width = 1
        cell_style = ':'

        fig = plt.figure(figsize=(fig_size, fig_aspect * fig_size))

        ax = fig.add_subplot(projection='3d')
        if print_nodes:
            ax.scatter(
                self.primal_nodes_positions[:, 0], self.primal_nodes_positions[:, 1], self.primal_nodes_positions[:, 2],
                s=node_size, c=primal_node_color)
            ax.scatter(
                self.dual_nodes_positions[:, 0], self.dual_nodes_positions[:, 1], self.dual_nodes_positions[:, 2],
                s=node_size, c=dual_node_color)
        if print_edges:
            for edge_plot in self.plotted_lattice_edges:
                ax.plot3D(*edge_plot.T, edge_color, alpha=edge_alpha, lw=edge_width)
        if print_cells:
            for cell_plot in self.cells_shapes:
                ax.plot3D(*cell_plot.T, cell_color, alpha=cell_alpha, lw=cell_width, ls=cell_style)
        plt.axis('off')
        plt.show()

    # Function to draw cells with odd syndromes in the lattice
    # TODO: For now it only works for primal cells, extend it also to dual

    def draw_odd_syndromes(self, syndrom_meas_list=None, error_qbts=None, matching_matrix=None, latt_type='primal',
                           print_nodes=False, print_edges=False, print_cells=True, print_error_qbts=True,
                           fig_size=10, fig_aspect=None):

        # syndrom_meas_list is a list of all syndromes,
        # with 0s for measured syndromes providing even parity (no error detected),
        #  and 1s for measured syndromes providing odd parity (error detected).

        # error_qbts is a list of which qubits had a Z error (used for a simple single qubit depolarizing noise model)

        if syndrom_meas_list is None:
            if error_qbts is not None:
                if matching_matrix is None:
                    matching_matrix = self.get_matching_matrix(latt_type)
                syndrom_meas_list = matching_matrix@error_qbts % 2
            else:
                raise ValueError('One between syndrom_meas_list or error_qbts needs to be provided')

        # Build edges of odd syndromes
        odd_syndr_shapes = []
        for cell_ix, is_odd in enumerate(syndrom_meas_list):
            if is_odd:
                this_cell_pos = self.cell_ix_to_xytcoords(cell_ix) + np.array((0.5, 0.5, 0.5))

                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift - self.yshift - self.tshift,
                                                  this_cell_pos + self.xshift - self.yshift - self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift - self.yshift + self.tshift,
                                                  this_cell_pos + self.xshift - self.yshift + self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift + self.yshift - self.tshift,
                                                  this_cell_pos + self.xshift + self.yshift - self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift + self.yshift + self.tshift,
                                                  this_cell_pos + self.xshift + self.yshift + self.tshift)))

                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift - self.yshift - self.tshift,
                                                  this_cell_pos - self.xshift + self.yshift - self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift - self.yshift + self.tshift,
                                                  this_cell_pos - self.xshift + self.yshift + self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos + self.xshift - self.yshift - self.tshift,
                                                  this_cell_pos + self.xshift + self.yshift - self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos + self.xshift - self.yshift + self.tshift,
                                                  this_cell_pos + self.xshift + self.yshift + self.tshift)))

                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift - self.yshift - self.tshift,
                                                  this_cell_pos - self.xshift - self.yshift + self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos - self.xshift + self.yshift - self.tshift,
                                                  this_cell_pos - self.xshift + self.yshift + self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos + self.xshift - self.yshift - self.tshift,
                                                  this_cell_pos + self.xshift - self.yshift + self.tshift)))
                odd_syndr_shapes.append(np.array((this_cell_pos + self.xshift + self.yshift - self.tshift,
                                                  this_cell_pos + self.xshift + self.yshift + self.tshift)))

        if fig_aspect is None:
            fig_aspect = self.lattice_t_size / ((self.lattice_x_size + self.lattice_y_size) / 2)

        node_size = 200 / (self.lattice_x_size * self.lattice_y_size)

        primal_node_color = 'black'
        dual_node_color = 'blue'

        edge_color = 'black'
        edge_alpha = 1
        edge_width = 1

        cell_color = 'black'
        cell_alpha = 0.5
        cell_width = 1
        cell_style = ':'

        oddsyndr_color = 'red'
        oddsyndr_alpha = 1
        oddsyndr_width = 1
        oddsyndr_style = '-'

        error_qbt_color = 'red'

        fig = plt.figure(figsize=(fig_size, fig_aspect * fig_size))

        ax = fig.add_subplot(projection='3d')
        if print_nodes:
            ax.scatter(
                self.primal_nodes_positions[:, 0], self.primal_nodes_positions[:, 1], self.primal_nodes_positions[:, 2],
                s=node_size, c=primal_node_color)
            ax.scatter(
                self.dual_nodes_positions[:, 0], self.dual_nodes_positions[:, 1], self.dual_nodes_positions[:, 2],
                s=node_size, c=dual_node_color)
        if print_edges:
            for edge_plot in self.plotted_lattice_edges:
                ax.plot3D(*edge_plot.T, edge_color, alpha=edge_alpha, lw=edge_width)
        if print_cells:
            for cell_plot in self.cells_shapes:
                ax.plot3D(*cell_plot.T, cell_color, alpha=cell_alpha, lw=cell_width, ls=cell_style)

        for odd_syndr_plot in odd_syndr_shapes:
            ax.plot3D(*odd_syndr_plot.T, oddsyndr_color, alpha=oddsyndr_alpha, lw=oddsyndr_width, ls=oddsyndr_style)

        if print_error_qbts and (error_qbts is not None):
            if latt_type == 'primal':
                error_nodes = self.primal_nodes_positions[error_qbts == 1]
            elif latt_type == 'dual':
                error_nodes = self.dual_nodes_positions[error_qbts == 1]
            else:
                raise ValueError('latt_type can be only "primal" or "dual"')
            ax.scatter(
                error_nodes[:, 0], error_nodes[:, 1], error_nodes[:, 2],
                s=node_size, c=error_qbt_color)

        plt.axis('off')
        plt.show()

    # Functions that build the matching matrix for syndromes of the lattice
    def get_matching_matrix(self, latt_type='primal'):
        if latt_type == 'primal':
            num_cells = self.num_primal_cells
            num_qubits = self.num_primal_qbts
            cells_struct = self.primal_cells_qbts_struct
        elif latt_type == 'dual':
            num_cells = self.num_dual_cells
            num_qubits = self.num_dual_qbts
            cells_struct = self.dual_cells_qbts_struct
        else:
            raise ValueError('latt_type can be only "primal" or "dual"')

        H_matrix = np.zeros((num_cells, num_qubits), dtype=np.uint8)
        for syndr_ix, syndr_nodes_list in enumerate(cells_struct):
            syndr_qubits = syndr_nodes_list[syndr_nodes_list != self.empty_ix]
            H_matrix[syndr_ix, syndr_qubits] = np.ones(len(syndr_qubits), dtype=np.uint8)
        return H_matrix





########################################################################################################################

##############################
###          MAIN          ###
##############################

if __name__ == '__main__':
    L_x = 4
    L_y = 4
    L_t = 4
    boundary_is_primal = ((True, True), (True, True), (True, True))
    # boundary_is_primal = ((False, False), (False, False), (False, False))

    Rauss_Lattice = RaussendorfLattice(L_x, L_y, L_t, boundary_is_primal)

    print('# prim qbts:', len(Rauss_Lattice.primal_nodes_positions), Rauss_Lattice.num_primal_qbts)
    print('# dual qbts:', len(Rauss_Lattice.dual_nodes_positions), Rauss_Lattice.num_dual_qbts)

    print('\n primal cells struct:')
    print(Rauss_Lattice.primal_cells_qbts_struct)

    print('\n primal cells matching matrix:')
    H_matrix = Rauss_Lattice.get_matching_matrix(latt_type='primal')
    print(H_matrix)


    # print('\n dual cells struct:')
    # print(Rauss_Lattice.dual_cells_qbts_struct)

    Rauss_Lattice.draw_lattice()

    # sample a random list of Pauli Z errors on the physical qubits and plot error qubits and syndroms
    pauli_Z_error_prob = 0.02
    qubit_error_list = np.random.binomial(1, pauli_Z_error_prob, Rauss_Lattice.num_primal_qbts)

    # qubit_error_list = np.zeros(20)
    # qubit_error_list[4] = 1
    # qubit_error_list[15] = 1

    print('\nqubit_error_list:', qubit_error_list)
    meas_syndroms_list = (H_matrix @ qubit_error_list) % 2
    print('\nmeas_syndroms_list:', meas_syndroms_list)

    Rauss_Lattice.draw_odd_syndromes(meas_syndroms_list, qubit_error_list)
