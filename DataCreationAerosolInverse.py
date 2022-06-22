from ImportFile import *

pi = math.pi


class DefineDataset:
    def __init__(self,
                 extrema_values,
                 parameters_values,
                 type_of_coll,
                 n_collocation,
                 n_boundary,
                 n_initial,
                 n_internal,
                 batches,
                 random_seed,
                 output_dimension,
                 space_dimensions=1,
                 time_dimensions=1,
                 parameter_dimensions=0,
                 obj=None,
                 shuffle=False,
                 type_point_param=None
                 ):
        self.extrema_values = extrema_values
        self.parameters_values = parameters_values
        self.type_of_coll = type_of_coll
        self.space_dimensions = space_dimensions
        self.time_dimensions = time_dimensions
        self.parameter_dimensions = parameter_dimensions
        self.output_dimension = output_dimension

        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.n_internal = n_internal

        self.batches = batches
        self.random_seed = random_seed
        self.data = None
        self.data_no_batches = None
        self.obj = obj
        if self.obj is not None:
            self.n_object = obj.n_object_space * obj.n_object_time
        else:
            self.n_object = 0
        self.n_samples = self.n_collocation + 2 * self.n_boundary * self.space_dimensions + self.n_initial * self.time_dimensions + self.n_internal + self.n_object
        print(self.time_dimensions, self.space_dimensions, self.parameter_dimensions)
        self.input_dimensions = self.time_dimensions + self.space_dimensions + self.parameter_dimensions
        self.BC = None
        self.shuffle = shuffle
        self.type_point_param = type_point_param

        if self.batches == "full":
            self.batches = int(self.n_samples)
        else:
            self.batches = int(self.batches)

    def assemble_dataset(self):

        fraction_coll = int(self.batches * self.n_collocation / self.n_samples)
        fraction_boundary = int(self.batches * 2 * self.n_boundary * self.space_dimensions / self.n_samples)
        fraction_initial = int(self.batches * self.n_initial / self.n_samples)
        fraction_ob = int(self.batches * self.n_object / self.n_samples)

        #############################################

        BC = list()

        x_b, y_b = Ec.add_boundary(2 * Ec.space_dimensions * self.n_boundary)
        x_coll, y_coll = Ec.add_collocations(self.n_collocation)
        x_internal, y_internal = Ec.add_internal_points(self.n_internal)
        fraction_internal = int(self.batches * self.n_internal / self.n_samples)

        # quit()
        if self.n_boundary == 0:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=1,
                                            shuffle=False)
        else:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=fraction_boundary,
                                            shuffle=self.shuffle)

        if self.n_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=1,
                                        shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll,
                                        shuffle=self.shuffle)

        if fraction_internal == 0 and fraction_initial == 0:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_internal, y_internal),
                                                    batch_size=1, shuffle=False)
        else:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_internal, y_internal),
                                                    batch_size=fraction_initial + fraction_internal, shuffle=self.shuffle)
        print("###################################")
        print(x_coll, y_coll.shape)
        print(x_internal, y_internal.shape)
        print(x_b, y_b.shape)
        print("###################################")

        if self.obj is not None:
            x_ob, y_ob, BC_ob = self.obj.construct_object()

            BC.append(BC_ob)

        self.BC = BC

    def generator_param_samples(self, samples, dim, random_seed):

        if self.type_point_param == "uniform":
            torch.random.manual_seed(random_seed)
            return torch.rand([samples, dim]).type(torch.FloatTensor)
        elif self.type_point_param == "normal":
            torch.random.manual_seed(random_seed)
            return torch.randn([samples, dim]).type(torch.FloatTensor)
        elif self.type_point_param == "sobol":
            skip = random_seed
            data = np.full((samples, dim), np.nan)
            for j in range(samples):
                seed = j + skip
                data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
            return torch.from_numpy(data).type(torch.FloatTensor)

    def transform_param_data(self, tensor):
        if self.type_point_param == "uniform" or self.type_point_param == "sobol":
            extrema_0 = self.parameters_values[:, 0]
            extrema_f = self.parameters_values[:, 1]
            return tensor * (extrema_f - extrema_0) + extrema_0
        elif self.type_point_param == "normal":
            mean = self.parameters_values[:, 0]
            std = self.parameters_values[:, 1]
            return tensor * std + mean
        else:
            raise ValueError()

    def generator_points(self, samples, dim, random_seed):
#inputs: samples, dim
        if self.type_of_coll == "random":
            torch.random.manual_seed(random_seed)
            return torch.rand([samples, dim]).type(torch.FloatTensor)
        elif self.type_of_coll == "lhs":
            return torch.from_numpy(lhs(dim, samples=samples, criterion='center')).type(torch.FloatTensor)
        elif self.type_of_coll == "grid":
            if dim == 2:
                ratio = (self.extrema_values[0, 1] - self.extrema_values[0, 0]) / (
                        self.extrema_values[1, 1] - self.extrema_values[1, 0])
                sqrt_samples_2 = int(np.sqrt(samples / ratio))
                dir2 = np.linspace(0, 1, sqrt_samples_2)
                sqrt_samples_1 = int(samples / sqrt_samples_2)
                dir1 = np.linspace(0, 1, sqrt_samples_1)
                return torch.from_numpy(
                    np.array([[x_i, y_i] for x_i in dir1 for y_i in dir2]).reshape(dir1.shape[0] * dir2.shape[0], 2)).type(torch.FloatTensor)
            else:
                return torch.linspace(0, 1, samples)
        elif self.type_of_coll == "sobol":
            skip = random_seed
            data = np.full((samples, dim), np.nan)
            for j in range(samples):
                seed = j + skip
                data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
            return torch.from_numpy(data).type(torch.FloatTensor)
