from ImportFile import *
from datetime import datetime
# from InverseRTEPINN import *

parameters_values = torch.tensor([[0, 2 * pi],
                                  [0, pi],
                                  [0, 1]])
parameter_dimensions = parameters_values.shape[0]
#TODO copied from the main
#S.Z size of the space domain vector x
space_dimensions = 1
time_dimensions = 0
#i think i need 3 dimensional outputs where they have only height and one angle
output_dimension = 1
assign_g = True
average = False
type_of_points = "sobol"
input_dimensions = 1
kernel_type = "isotropic"
n_quad = 10
#end copied here

def generator_domain_samples(points, boundary):
    if boundary:
        n_single_dim = int(points.shape[0] / input_dimensions)
        print(n_single_dim)
        for i in range(input_dimensions):
            n = int(n_single_dim / 2)
            points[2 * i * n:n * (2 * i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=0.0)
            points[n * (2 * i + 1):2 * n * (i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=1.0)

    return points

def generator_param_samples(points):
    extrema_0 = parameters_values[:2, 0]
    extrema_f = parameters_values[:2, 1]
    points = points * (extrema_f - extrema_0) + extrema_0
    s = torch.tensor(()).new_full(size=(points.shape[0], 3), fill_value=0.0)
    phi = points[:, 0]
    theta = points[:, 1]
    s[:, 0] = torch.cos(phi) * torch.sin(theta)
    s[:, 1] = torch.sin(phi) * torch.sin(theta)
    s[:, 2] = torch.cos(theta)
    return s


def get_points(samples, dim, type_point_param, random_seed):
    if type_point_param == "uniform":
        torch.random.manual_seed(random_seed)
        points = torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_point_param == "sobol":
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        points = torch.from_numpy(data).type(torch.FloatTensor)
    return points

#This function creates the collocation points, which are the points for the scattering integral
def add_collocations(n_collocation):
    n_coll_int = int(3 / 4 * n_collocation)
    print("Adding Collocation")
    points = get_points(n_collocation, 5, "sobol", 16)
    dom_int = points[:n_coll_int, :3]
    angles_int = points[:n_coll_int, 3:]
    dom_b = points[n_coll_int:, :3]
    angles_b = points[n_coll_int:, 3:]
    dom = torch.cat([generator_domain_samples(dom_int, boundary=False), generator_domain_samples(dom_b, boundary=True)])
    s = torch.cat([generator_param_samples(angles_int), generator_param_samples(angles_b)])

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)

    return torch.cat([x, y, z, s], 1), u



def initialize_inputs(len_sys_argv):
    if len_sys_argv == 1:
        # Random Seed for sampling the dataset
        sampling_seed_ = 32

        # Number of training+validation points
        n_coll_ = 8192 #TODO what is that
        n_u_ = 120 #TODO what is that
        n_int_ = 0 #TODO what is that

        # Additional Info
        folder_path_ = "Inverse"
        point_ = "sobol"
        validation_size_ = 0.0
        network_properties_ = {
            "hidden_layers": 4,
            "neurons": 20,
            "residual_parameter": 1,
            "kernel_regularizer": 2,
            "regularization_parameter": 0,
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 1,
            "activation": "tanh"
        }
        retrain_ = 32
        shuffle_ = False
    else:
        raise ValueError("One input is missing")

    return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, point_, validation_size_, network_properties_, retrain_, shuffle_

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
        # fraction_ob = int(self.batches * self.n_object / self.n_samples)

        #############################################
        #TODO boundry condition?
        BC = list()

        x_b, y_b = add_boundary(2 * space_dimensions * self.n_boundary)
        x_coll, y_coll = add_collocations(self.n_collocation)
        x_internal, y_internal = add_internal_points(self.n_internal)
        #TODO this variable probably should be deleted
        x_time_internal = x_internal
        y_time_internal = y_internal
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
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                    batch_size=1, shuffle=False)
        else:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                    batch_size=fraction_initial + fraction_internal, shuffle=self.shuffle)
        print("###################################")
        print(x_coll, y_coll.shape)
        print(x_time_internal, y_time_internal.shape)
        print(x_b, y_b.shape)
        print("###################################")
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

# def add_internal_points(n_internal):
#     n_int = int(n_internal * 3 / 4)
#     print("Adding Internal Points")
#     points = get_points(n_internal, 5, "uniform", 16)
#     dom_int = points[:n_int, :3]
#     angles_int = points[:n_int, 3:]
#     dom_b = points[n_int:, :3]
#     angles_b = points[n_int:, 3:]
#     dom = torch.cat([generator_domain_samples(dom_int, boundary=False), generator_domain_samples(dom_b, boundary=True)])
#     s = torch.cat([generator_param_samples(angles_int), generator_param_samples(angles_b)])
#
#     x = dom[:, 0].reshape(-1, 1)
#     y = dom[:, 1].reshape(-1, 1)
#     z = dom[:, 2].reshape(-1, 1)
#
#     u = exact(x.reshape(-1, ), y.reshape(-1, ), z.reshape(-1, ), s[:, 0], s[:, 1], s[:, 2]).reshape(-1, 1)
#     g = G(x.reshape(-1, ), y.reshape(-1, ), z.reshape(-1, )).reshape(-1, 1)
#
#     if assign_g:
#         return torch.cat([x, y, z, s], 1), g
#     else:
#         return torch.cat([x, y, z, s], 1), u
