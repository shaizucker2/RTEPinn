from ImportFile import *

pi = math.pi
#TODO make sure this is the best place to put this
space_dimensions = 1
time_dimensions = 0
output_dimension = 1
assign_g = True
average = False
type_of_points = "sobol"
r_min = 0.0
input_dimensions = 1
kernel_type = "isotropic"
n_quad = 10

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")

class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        # pre-activation
        return x * torch.sigmoid(x)


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['swish']:
        return Swish()
    else:
        raise ValueError('Unknown activation function')


class Pinns(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties, additional_models=None,
                 solid_object=None):
        super(Pinns, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.regularization_param = float(network_properties["regularization_parameter"])
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.solid_object = solid_object #TODO delete this param
        self.additional_models = additional_models #TODO what is this

        self.activation = activation(self.act_string)
        # self.a = nn.Parameter(torch.randn((1,)))

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for l in self.hidden_layers:
            x = self.activation(l(x))
        return self.output_layer(x)


def fit(model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, training_set_class, validation_set_clsss=None, verbose=False, training_ic=False):
    num_epochs = model.num_epochs

    train_losses = list([np.NAN])
    val_losses = list()
    freq = 50 #TODO what this does?

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal
    # mo
    model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if epoch < epoch_ADAM:
            print("Using ADAM")
            optimizer = optimizer_ADAM
        else:
            print("Using LBFGS")
            optimizer = optimizer_LBFGS
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        print(len(training_boundary))
        print(len(training_coll))
        print(len(training_initial_internal))

        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_), (x_u_train_, u_train_)) in enumerate(
                    zip(training_coll, training_boundary, training_initial_internal)):
                if verbose and epoch % freq == 0:
                    print("Batch Number:", step)

                x_ob = None
                u_ob = None

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = CustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    # print(train_losses[0])
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
        elif len(training_boundary) == 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_initial_internal)):

                x_ob = None
                u_ob = None
                #S.Z changed to 2 2 was 4
                x_b_train_ = torch.full((4, 1), 0)
                u_b_train_ = torch.full((4, 1), 0)

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = CustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    # print(train_losses[0])
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
        elif len(training_boundary) != 0 and len(training_initial_internal) == 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):

                x_ob = None
                u_ob = None

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    x_u_train_ = x_u_train_.cuda()
                    u_train_ = u_train_.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = CustomLoss()(model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, x_ob, u_ob,
                                          training_set_class, training_ic)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        if validation_set_clsss is not None:

            N_coll_val = validation_set_clsss.n_collocation
            N_b_val = validation_set_clsss.n_boundary
            validation_set = validation_set_clsss.data_no_batches

            for x_val, y_val in validation_set:
                model.eval()

                x_coll_val = x_val[:N_coll_val, :]
                x_b_val = x_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val, :]
                u_b_val = y_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val]
                x_u_val = x_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val:, :]
                u_val = y_val[N_coll_val:N_coll_val + 2 * validation_set_clsss.space_dimensions * N_b_val:, :]

                if torch.cuda.is_available():
                    x_coll_val = x_coll_val.cuda()
                    x_b_val = x_b_val.cuda()
                    u_b_val = u_b_val.cuda()
                    x_u_val = x_u_val.cuda()
                    u_val = u_val.cuda()

                loss_val = CustomLoss()(model, x_u_val, u_val, x_b_val, u_b_val, x_coll_val, validation_set_clsss)

                if torch.cuda.is_available():
                    del x_coll_val
                    del x_b_val
                    del u_b_val
                    del x_u_val
                    del u_val
                    torch.cuda.empty_cache()

                    # val_losses.append(loss_val)
                if verbose and epoch % 100 == 0:
                    print("Validation Loss:", loss_val)

    history = [train_losses, val_losses] if validation_set_clsss is not None else [train_losses]

    return train_losses[0]


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, x_obj, u_obj, dataclass,
                training_ic, computing_error=False):

        lambda_residual = network.lambda_residual
        lambda_reg = network.regularization_param
        order_regularizer = network.kernel_regularizer
        space_dimensions = dataclass.space_dimensions
        BC = dataclass.BC
        solid_object = dataclass.obj #TODO this will always be none

        if x_b_train.shape[0] <= 1:
            space_dimensions = 0

        u_pred_var_list = list()
        u_train_var_list = list()
        for j in range(dataclass.output_dimension):

            # Space dimensions
            # Not used for the radiative project, check at the else part
            u_pred_b, u_train_b = apply_BC(x_b_train, u_b_train, network)
            u_pred_var_list.append(u_pred_b)
            u_train_var_list.append(u_train_b)

            # Time Dimension
            if x_u_train.shape[0] != 0:
                # This is used to solve the radiative inverse problem
                try:
                    if j == 0:
                        if assign_g:
                            # print("Assign G")
                            g = get_G(network, x_u_train[:, :3], n_quad)
                            u_pred_var_list.append(g)

                        else:
                            # print("Not assign G")
                            u_pred_var_list.append(network(x_u_train)[:, j])
                        u_train_var_list.append(u_train[:, j])
                    if j == 1:
                        # print(x_b_train)
                        phys_coord_b = x_b_train[:, :3]
                        if average:
                            u_pred_var_list.append(get_average_inf_q(network, phys_coord_b, 10))
                        else:
                            u_pred_var_list.append(network(x_b_train)[:, j])
                        u_train_var_list.append(K(phys_coord_b[:, 0], phys_coord_b[:, 1], phys_coord_b[:, 2]))
                        # Compute tikonov regularization

                        x_f_train.requires_grad = True
                        k = network(x_f_train)[:, j].reshape(-1, )
                        lambda_k = 0.01

                        grad_k = torch.autograd.grad(k, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0]

                        grad_k_x = grad_k[:, 0]
                        grad_k_y = grad_k[:, 1]

                        tik_reg = torch.sqrt(lambda_k * torch.mean(grad_k_x ** 2 + grad_k_y ** 2)).reshape(1, )

                        u_pred_var_list.append(tik_reg)
                        u_train_var_list.append(torch.zeros_like(tik_reg))
                except AttributeError:
                    u_pred_var_list.append(network(x_u_train)[:, j])
                    u_train_var_list.append(u_train[:, j])


        u_pred_tot_vars = torch.cat(u_pred_var_list, 0)
        u_train_tot_vars = torch.cat(u_train_var_list, 0)

        if not computing_error and torch.cuda.is_available():
            u_pred_tot_vars = u_pred_tot_vars.cuda()
            u_train_tot_vars = u_train_tot_vars.cuda()

        assert not torch.isnan(u_pred_tot_vars).any()

        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** 2))

        if not training_ic:

            res = compute_res(network, x_f_train, space_dimensions, solid_object, computing_error)
            res_train = torch.tensor(()).new_full(size=(res.shape[0],), fill_value=0.0)

            if not computing_error and torch.cuda.is_available():
                res = res.cuda()
                res_train = res_train.cuda()

            loss_res = (torch.mean(abs(res) ** 2))

            u_pred_var_list.append(res)
            u_train_var_list.append(res_train)

        loss_reg = regularization(network, order_regularizer)
        if not training_ic:
            loss_v = torch.log10(
                loss_vars + lambda_residual * loss_res + lambda_reg * loss_reg)  # + lambda_reg/loss_reg
        else:
            loss_v = torch.log10(loss_vars + lambda_reg * loss_reg)
        print("final loss:", loss_v.detach().cpu().numpy().round(4), " ", torch.log10(loss_vars).detach().cpu().numpy().round(4), " ",
              torch.log10(loss_res).detach().cpu().numpy().round(4))
        return loss_v


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            # gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.0)

    model.apply(init_weights)


def compute_error(dataset, trained_model):
    training_coll = dataset.data_coll
    training_boundary = dataset.data_boundary
    training_initial_internal = dataset.data_initial_internal
    error_mean = 0
    n = 0
    if len(training_boundary) != 0 and len(training_initial_internal) != 0:
        for step, ((x_coll, u_coll_train_), (x_b, u_b), (x_u, u)) in enumerate(
                zip(training_coll, training_boundary, training_initial_internal)):
            x_ob = None
            u_ob = None
            loss = CustomLoss()(trained_model, x_u, u, x_b, u_b, x_coll, x_ob, u_ob, dataset, False, True)
            error = torch.sqrt(10 ** loss)
            error_mean = error_mean + error
            n = n + 1
        error_mean = error_mean / n
    if len(training_boundary) != 0 and len(training_initial_internal) == 0:
        for step, ((x_coll, u_coll_train_), (x_b, u_b)) in enumerate(
                zip(training_coll, training_boundary)):
            x_ob = None
            u_ob = None
            x_u = torch.full((0, 1), 0)
            u = torch.full((0, 1), 0)
            loss = CustomLoss()(trained_model, x_u, u, x_b, u_b, x_coll, x_ob, u_ob, dataset, False, True)
            error = torch.sqrt(10 ** loss)
            error_mean = error_mean + error
            n = n + 1
        error_mean = error_mean / n
    if len(training_boundary) == 0 and len(training_initial_internal) != 0:
        for step, ((x_coll, u_coll_train_), (x_u, u)) in enumerate(
                zip(training_coll, training_initial_internal)):
            x_ob = None
            u_ob = None
            x_b = torch.full((0, 1), 0)
            u_b = torch.full((0, 1), 0)

            loss = CustomLoss()(trained_model, x_u, u, x_b, u_b, x_coll, x_ob, u_ob, dataset, False, True)
            error = torch.sqrt(10 ** loss)
            error_mean = error_mean + error
            n = n + 1
        error_mean = error_mean / n
    return error_mean


def compute_error_nocoll(dataset, trained_model):
    training_initial_internal = dataset.data_initial_internal
    error_mean = 0
    n = 0
    for step, (x_u, u) in enumerate(training_initial_internal):
        loss = StandardLoss()(trained_model, x_u, u)
        error = torch.sqrt(10 ** loss)
        error_mean = error_mean + error
        n = n + 1
    error_mean = error_mean / n
    return error_mean


class StandardLoss(torch.nn.Module):
    def __init__(self):
        super(StandardLoss, self).__init__()

    def forward(self, network, x_u_train, u_train):
        loss_reg = regularization(network, 2)
        lambda_reg = network.regularization_param
        u_pred = network(x_u_train)

        x_u_train.requires_grad = True
        u = network(x_u_train).reshape(-1, )
        u_ex = torch.tensor(9.) / torch.cosh(np.sqrt(3.) / 2 * (x_u_train[:, 1] - 3 * x_u_train[:, 0])) ** 2
        grad_u = torch.autograd.grad(u, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0], ), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        grad_u_xxx = torch.autograd.grad(grad_u_xx, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        plt.scatter(x_u_train[:, 1].detach().numpy(), u.detach().numpy(), s=10)
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_x.detach().numpy(), s=10)
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xx.detach().numpy(), s=10)
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xxx.detach().numpy(), s=10)

        grad_u = torch.autograd.grad(u_ex, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0], ), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        grad_u_xxx = torch.autograd.grad(grad_u_xx, x_u_train, grad_outputs=torch.ones(x_u_train.shape[0]), create_graph=True)[0][:, 1]
        plt.scatter(x_u_train[:, 1].detach().numpy(), u_ex.detach().numpy(), s=10, marker="v")
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_x.detach().numpy(), s=10, marker="v")
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xx.detach().numpy(), s=10, marker="v")
        plt.scatter(x_u_train[:, 1].detach().numpy(), grad_u_xxx.detach().numpy(), s=10, marker="v")
        plt.pause(0.0000001)
        plt.clf()
        loss = torch.log10(torch.mean((u_train - u_pred) ** 2) + lambda_reg * loss_reg)
        del u_train, u_pred
        print(loss)
        return loss


def StandardFit(model, optimizer_ADAM, optimizer_LBFGS, training_set_class, validation_set_clsss=None, verbose=False):
    num_epochs = model.num_epochs

    train_losses = list([np.nan])
    val_losses = list()
    freq = 4

    model.train()
    training_initial_internal = training_set_class.data_initial_internal
    epoch_LSBGF = num_epochs
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if epoch < epoch_LSBGF:
            print("Using LSBGF")
            optimizer = optimizer_LBFGS
        else:
            print("Using Adam")
            optimizer = optimizer_ADAM
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        for step, (x_u_train_, u_train_) in enumerate(training_initial_internal):
            if verbose and epoch % freq == 0:
                print("Batch Number:", step)

            if torch.cuda.is_available():
                x_u_train_ = x_u_train_.cuda()
                u_train_ = u_train_.cuda()

            def closure():
                optimizer.zero_grad()
                loss_f = StandardLoss()(model, x_u_train_, u_train_)
                loss_f.backward()
                train_losses[0] = loss_f
                return loss_f

            optimizer.step(closure=closure)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del x_u_train_
            del u_train_

    history = [train_losses, val_losses] if validation_set_clsss is not None else [train_losses]

    return train_losses[0]


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error):
    x_f_train.requires_grad = True
    x = x_f_train[:, 0]
    y = x_f_train[:, 1]
    z = x_f_train[:, 2]
    s = x_f_train[:, 3:]

    phys_coord = x_f_train[:, :3]

    u = network(x_f_train)[:, 0].reshape(-1, )
    if average:
        absorb = get_average_inf_q(network, phys_coord, n_quad)
    else:
        absorb = network(x_f_train)[:, 1].reshape(-1, )

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0]

    grad_u_x = grad_u[:, 0]
    grad_u_y = grad_u[:, 1]
    grad_u_z = grad_u[:, 2]
    s1 = s[:, 0]
    s2 = s[:, 1]
    s3 = s[:, 2]

    scatter_kernel = compute_scatter(x_f_train, network)

    res = s1 * grad_u_x + s2 * grad_u_y + s3 * grad_u_z + (absorb + S(x, y, z)) * u - f(x, y, z, s1, s2, s3) - S(x, y, z) * scatter_kernel

    return res











