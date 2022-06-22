from ImportFile import *
import DataCreation as dc
from datetime import datetime
pi = math.pi
# torch.manual_seed(42)

from ImportFile import *
from datetime import datetime
pi = math.pi
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




sampling_seed, N_coll, N_u, N_int, folder_path, point, validation_size, network_properties, retrain, shuffle = dc.initialize_inputs(len(sys.argv))
#also not in the one dimentional case

extrema = None
space_dimensions = 3
#TODO get rid of this
try:
    parameters_values = dc.parameters_values
    parameter_dimensions = parameters_values.shape[0]
    type_point_param = "sobol"
except AttributeError:
    print("No additional parameter found")
    parameters_values = None
    parameter_dimensions = 0
    type_point_param = None
#TODO what is paramter dimension?
input_dimensions = parameter_dimensions + space_dimensions
# since it is an output as a function of x and mu so a matrix essintially, TODO go back to make sure and get ready for
# the addition of more dimensions in the polzrized case
output_dimension = 2
#I think the second output dimension is the k

print(input_dimensions)
#TODO understand what's mode
mode = "none"
max_iter = 50000
if network_properties["epochs"] != 1:
    max_iter = 1


#S.Z it seems they take train points out of the entire givan point to assess the accuracy of the algorithm
N_u_train = int(N_u * (1 - validation_size))
N_coll_train = int(N_coll * (1 - validation_size))
N_int_train = int(N_int * (1 - validation_size))
N_object_train = 0 #TODO try to delete this variable
N_train = N_u_train + N_coll_train + N_int_train + N_object_train

N_u_val = N_u - N_u_train
N_coll_val = N_coll - N_coll_train
N_int_val = N_int - N_int_train
# N_object_val = N_object - N_object_train TODO delete row
N_val = N_u_val + N_coll_val + N_int_val
#TODO understand what the hell happens here
N_b_train = int(N_u_train / (2 * space_dimensions))
N_i_train = 0


print("\n######################################")
print("*******Info Training Points********")
print("Number of train collocation points: ", N_coll_train)
print("Number of initial and boundary points: ", N_u_train, N_i_train, N_b_train)
print("Number of internal points: ", N_int_train)
print("Total number of training points: ", N_train)

print("\n######################################")
print("*******Info Validation Points********")
print("Number of train collocation points: ", N_coll_val)
print("Number of initial and boundary points: ", N_u_val)
print("Number of internal points: ", N_int_val)
print("Total number of training points: ", N_val)

print("\n######################################")
print("*******Network Properties********")
pprint.pprint(network_properties)
batch_dim = network_properties["batch_size"]

print("\n######################################")
print("*******Parameter Dimension********")
print(parameter_dimensions)

if batch_dim == "full":
    batch_dim = N_train

# ##############################################################################################
# Datasets Creation
print("DIMENSION")
print(space_dimensions, 0, parameter_dimensions)
training_set_class = dc.DefineDataset(extrema,
                                   parameters_values,
                                   point,
                                   N_coll_train,
                                   N_b_train,
                                   N_i_train,
                                   N_int_train,
                                   batches=batch_dim,
                                   output_dimension=output_dimension,
                                   space_dimensions=space_dimensions,
                                   time_dimensions=0,
                                   parameter_dimensions=parameter_dimensions,
                                   random_seed=sampling_seed,
                                   obj=None,
                                   shuffle=shuffle,
                                   type_point_param=type_point_param)

training_set_class.assemble_dataset()
training_set_no_batches = training_set_class.data_no_batches

validation_set_class = None

additional_models = None

model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
              network_properties=network_properties, additional_models=additional_models)
torch.manual_seed(retrain)
init_xavier(model)
if torch.cuda.is_available():
    print("Loading model on GPU")
    model.cuda()

start = time.time()
print("Fitting Model")
model.train()
epoch_ADAM = model.num_epochs - 1

# ##############################################################################################
# Model Training
optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)  # 1.0 * np.finfo(float).eps
optimizer_ADAM = optim.Adam(model.parameters(), lr=0.00005)
if N_coll_train != 0:
    final_error_train = fit(model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, training_set_class, validation_set_clsss=validation_set_class, verbose=True,
                            training_ic=False)
else: #TODO when running it didn't enter else, please delete!
    final_error_train = StandardFit(model, optimizer_ADAM, optimizer_LBFGS, training_set_class, validation_set_clsss=validation_set_class, verbose=True)
end = time.time() - start

print("\nTraining Time: ", end)

model = model.eval()
final_error_train = float(((10 ** final_error_train) ** 0.5).detach().cpu().numpy())
print("\n################################################")
print("Final Training Loss:", final_error_train)
print("################################################")
final_error_val = None
final_error_test = 0

# ##############################################################################################
# Plotting ang Assessing Performance
# time_str = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
# current_folder = folder_path +'\\' + time_str
# images_path = current_folder + "\\Images"
# os.mkdir(current_folder)
# os.mkdir(images_path)
# model_path = current_folder + "\\TrainedModel"
# os.mkdir(model_path)
#
# L2_test, rel_L2_test = Ec.compute_generalization_error(model, extrema, images_path)
Ec.plotting(model, 'Temp', extrema, None)
# Ec.plotting(model, images_path, extrema, solid_object)

end_plotting = time.time() - end

print("\nPlotting and Computing Time: ", end_plotting)
time_str = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
current_folder = folder_path +'\\' + time_str
images_path = current_folder + "\\Images"
os.mkdir(current_folder)
os.mkdir(images_path)
model_path = current_folder + "\\TrainedModel"
os.mkdir(model_path)
torch.save(model, model_path + "/model.pkl")
with open(model_path + os.sep + "Information.csv", "w") as w:
    keys = list(network_properties.keys())
    vals = list(network_properties.values())
    w.write(keys[0])
    for i in range(1, len(keys)):
        w.write("," + keys[i])
    w.write("\n")
    w.write(str(vals[0]))
    for i in range(1, len(vals)):
        w.write("," + str(vals[i]))

# with open(folder_path + '/InfoModel.txt', 'w') as file:
#     file.write("Nu_train,"
#                "Nf_train,"
#                "Nint_train,"
#                "validation_size,"
#                "train_time,"
#                "L2_norm_test,"
#                "rel_L2_norm,"
#                "error_train,"
#                "error_val,"
#                "error_test\n")
#     file.write(str(N_u_train) + "," +
#                str(N_coll_train) + "," +
#                str(N_int_train) + "," +
#                str(validation_size) + "," +
#                str(end) + "," +
#                str(L2_test) + "," +
#                str(rel_L2_test) + "," +
#                str(final_error_train) + "," +
#                str(final_error_val) + "," +
#                str(final_error_test))
