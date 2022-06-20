import math
from datetime import datetime

import numpy as np
import torch
from scipy.optimize import nnls
from torch import nn, optim

from benchmarking import writeMatrix
from colab.converter import main_format
from colab.geditt import run_gedit_pre1
from gedit_preprocess.MatrixTools import remove0s, qNormMatrices, getSharedRows, RescaleRows
from gedit_preprocess.getSigGenesModal import returnSigMatrix
from layers.unsuper_layer import EPSILON
from layers.unsuper_net import UnsuperNet
from layers.unsuper_net_new import UnsuperNetNew

inf = math.inf


def tensoring(X):
    # conver numpy array to torch tensor
    return torch.from_numpy(X).float()


def cost_tns(v, w, h, l_1=0, l_2=0):
    # util.cost_tns(data.v_train.tns,data.w.tns,data.h_train.tns)
    d = v - h.mm(w)
    return (0.5 * torch.pow(d, 2).sum() + l_1 * h.sum() + 0.5 * l_2 * torch.pow(h, 2).sum()) / h.shape[0]


def train_unsupervised(
    v_train,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.005,
    l_1=0,
    l_2=0,
    include_reg=True,
    ref_data=None,
):
    samples, features = v_train.shape
    h_0_train = tensoring(np.ones((samples, n_components)))
    w_init = tensoring(np.ones((n_components, features)) / features)
    # build the architicture
    if include_reg:
        deep_nmf = UnsuperNet(num_layers, n_components, features, l_1, l_2)
    else:
        deep_nmf = UnsuperNet(num_layers, n_components, features, 0, 0)

    for w in deep_nmf.parameters():
        w.data.fill_(0.1)

    if ref_data is not None:
        deep_nmf_params = list(deep_nmf.parameters())
        for w_index in range(len(deep_nmf_params)):
            w = deep_nmf_params[w_index]
            if w_index == 0:
                w.data = torch.from_numpy(np.dot(ref_data.T, ref_data)).float()
                # deep_nmf_params[w_index].requires_grad = False
            elif w_index == 1:
                w.data = ref_data.T
                # deep_nmf_params[w_index].requires_grad = False

        h_0_train = tensoring(
            np.asanyarray(
                [np.random.dirichlet([1 for i in range(n_components)]) for i in range(samples)], dtype=np.float
            )
        )

    # initialize parameters
    dnmf_w = w_init

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
    # Train the Network
    inputs = (h_0_train, v_train)
    dnmf_train_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        out = out / torch.clamp(out.sum(axis=1)[:, None], min=1e-12)  # maybe you can insert into the net
        # out = tensoring(np.where(out.sum(axis=1)[:, None] != 0, (out / out.sum(axis=1)[:, None]).detach().numpy(), 0))

        loss = cost_tns(v_train, dnmf_w, out, l_1, l_2)

        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # keep weights positive after gradient decent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=inf)

        # NNLS
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
        w_arrays = [nnls(out.data.numpy(), v_train.detach().numpy().T[f])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

        dnmf_train_cost.append(loss.item())

    return deep_nmf, dnmf_train_cost, dnmf_w, out


def train_unsupervised_new(
    v_train,
    num_layers,
    network_train_iterations,
    n_components,
    ref_path,
    mix_object,
    output_folder,
    verbose=True,
    lr=0.005,
    l_1=0,
    l_2=0,
    include_reg=True,
    ref_data=None,
    w1_opt=None,
    dist_mix_i=None,
):
    print(f'start time: {datetime.now().strftime("%d-%m, %H:%M:%S")}')
    samples, features = v_train.shape
    h_0_train = tensoring(np.ones((samples, n_components)))
    w_init = tensoring(np.ones((n_components, features)) / features)
    if include_reg:
        deep_nmf = UnsuperNetNew(num_layers, n_components, features, l_1, l_2)
    else:
        deep_nmf = UnsuperNetNew(num_layers, n_components, features, 0, 0)

    for w in deep_nmf.parameters():
        w.data = (2 + torch.randn(w.data.shape, dtype=w.data.dtype)) * np.sqrt(2.0 / w.data.shape[0]) * 150
    # for w in deep_nmf.parameters():
    #     w.data.fill_(0.1)

    h_0_train = tensoring(
        np.asanyarray(
            [np.random.dirichlet(np.random.randint(1, 20, size=n_components)) for i in range(samples)], dtype=np.float
        )
    )

    w0_init = torch.from_numpy(np.dot(ref_data.T, ref_data)).float()
    w1_init = ref_data.T

    deep_nmf_params = list(deep_nmf.parameters())
    if ref_data is not None:
        for w_index in range(len(deep_nmf_params)):
            w = deep_nmf_params[w_index]
            if w_index == 0:
                w.data = torch.from_numpy(np.dot(ref_data.T, ref_data)).float()
            elif w_index == 1:
                w.data = ref_data.T

        h_0_train = [nnls(ref_data.detach().numpy(), v_train.data.numpy()[kk])[0] for kk in range(len([5]))]
        h_0_train = np.asanyarray([d / sum(d) for d in h_0_train])
        h_0_train = torch.from_numpy(h_0_train).float()

    # initialize parameters
    dnmf_w = w_init

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
    # train_with_generated_data(ref_data, mix_object, deep_nmf, optimizerADAM)
    # train_with_generated_data(ref_data, ref_path, mix_object, output_folder, deep_nmf, optimizerADAM)
    # Train the Network
    inputs = (h_0_train, v_train)
    dnmf_train_cost = []
    torch.autograd.set_detect_anomaly(True)
    for i in range(network_train_iterations):
        out, h_list = deep_nmf(*inputs)
        # softmax = nn.Softmax(1)
        # out = softmax(out)
        # out = out / (torch.clamp(out.sum(axis=1)[:, None], min=1e-12) + EPSILON) # maybe you can insert into the net
        # out = tensoring(np.where(out.sum(axis=1)[:, None] != 0, (out / out.sum(axis=1)[:, None]).detach().numpy(), 0))

        # keep weights positive after gradient decent

        # for w in deep_nmf.parameters():
        #    w.data = w.clamp(min=0, max=inf)
        deep_nmf_params = list(deep_nmf.parameters())

        w_i = deep_nmf_params[1].T
        # w_i = deep_nmf_params[len(deep_nmf_params) - 1].T
        for j in range(len(h_list)):
            w_i = new_w(h_list[j].T, w_i, v_train.T)
        # dnmf_w = deep_nmf_params[len(deep_nmf_params)-1]
        if w1_opt == "1":
            dnmf_w = deep_nmf_params[1]
        elif w1_opt == "last":
            dnmf_w = deep_nmf_params[len(deep_nmf_params) - 1]
        elif w1_opt == "algo":
            dnmf_w = nn.Softmax(1)(w_i.T)
        else:
            w_arrays = [nnls(out.data.numpy(), v_train.detach().numpy().T[f])[0] for f in range(features)]
            nnls_w = np.stack(w_arrays, axis=-1)
            dnmf_w = torch.from_numpy(nnls_w).float()

        loss = cost_tns(v_train, dnmf_w, out, l_1, l_2)
        criterion = nn.MSELoss(reduction="mean")
        loss2 = torch.sqrt(criterion(out, dist_mix_i))

        out34 = out / (torch.clamp(out.sum(axis=1)[:, None], min=1e-12))
        loss34 = torch.sqrt(criterion(out34, dist_mix_i))

        if verbose:
            if i % 1000 == 0:
                print(f"i = {i}, loss =  {loss.item()},  loss criterion = {loss2} loss34 is {loss34}")
                if i > 100 and loss34.item() > 0.25 and loss34.item() < 2:
                    print("break")
                    return deep_nmf, dnmf_train_cost, dnmf_w, out
        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=inf)

        # w_arrays = [nnls(out.data.numpy(), v_train.detach().numpy().T[f])[0] for f in range(features)]
        # next_w =
        # w_arrays = [new_w(h_list[h_index], dnmf_w[h_index], v_train.detach().numpy().T) for h_index in range(len(h_list))]
        # nnls_w = np.stack(w_arrays, axis=-1)
        # dnmf_w = torch.from_numpy(nnls_w).float()

        dnmf_train_cost.append(loss.item())

    return deep_nmf, dnmf_train_cost, dnmf_w, out


def new_w(Hi, Wi, V):
    denominator = torch.add(Wi.matmul(Hi).matmul(Hi.T), EPSILON)
    numerator = V.matmul(Hi.T)
    delta = torch.div(numerator, denominator)
    return torch.mul(delta, Wi)


def run_nmf_on_data_data(data_file, signature, output_path, deep_nmf, reformat_path=None, y_value=None):
    signature_data = signature[1:, 1:].astype(float).T
    samples, genes_count = data_file.shape
    cells_count, genes_count = signature_data.shape

    H_init_np = np.ones((samples, cells_count))

    H_init = torch.from_numpy(H_init_np).float()
    data_torch = torch.from_numpy(data_file).float()
    out = deep_nmf(H_init, data_torch)
    normalize_out = out / out.sum(axis=1)[:, None]

    criterion = nn.MSELoss(reduction="mean")
    if y_value is not None:
        # ref_object_formated_data =
        loss = torch.sqrt(criterion(normalize_out, y_value))
        return loss
    result = np.zeros((out.shape[0] + 1, out.shape[1] + 1), dtype="U28")
    result[0, 1:] = signature[0][1:]
    result[1:, 1:] = normalize_out.detach().numpy()

    result[:, 0] = [f"Mixture_{i}" for i in range(len(result[:, 0]))]
    use_t = True
    if reformat_path is not None:
        ref_object_formated = main_format(result, reformat_path)
        result = ref_object_formated
        use_t = False
    writeMatrix(result, output_path, use_t)


def create_mix_train(mix_data, train_data):
    genes_names_in_mix = [g[0] for g in mix_data[1:]]

    result = np.zeros((mix_data.shape[0] - 1, mix_data.shape[1] - 1), dtype=float)
    for gene_index in range(1, len(train_data)):
        gene_name = train_data[gene_index][0]
        if gene_name in genes_names_in_mix:
            sig_index = genes_names_in_mix.index(gene_name)
            result[sig_index] = train_data[gene_index][1:]

    return result


def read_dataset_data(file_data):
    data_file = []
    for line in file_data:
        splitLine = line.strip().split("\t")
        if len(splitLine) == 1:
            splitLine = line.strip().split(",")
        data_file.append(splitLine)
    data_file2 = np.array(data_file)
    return data_file2


def generate_dists(signature_data, std, alpha=1):
    dist = np.asanyarray(
        [np.random.dirichlet([alpha for i in range(signature_data.shape[0])]) for i in range(100)], dtype=float
    )

    t = dist.dot(signature_data)
    t += np.random.normal(0, std, t.shape)
    t = np.maximum(t, 0)
    return t, dist


def train_supervised_one_sample(
    v_train, h_train, network_train_iterations, deep_nmf, optimizer, verbose=False, print_every=100
):
    n_h_rows, n_components = h_train.shape
    H_init_np = np.ones((n_h_rows, n_components))
    H_init = torch.from_numpy(H_init_np).float()

    criterion = nn.MSELoss(reduction="mean")
    # Train the Network
    loss_values = []
    for i in range(network_train_iterations):
        out = deep_nmf(H_init, v_train)
        loss = torch.sqrt(criterion(out, h_train))  # loss between predicted and truth

        if verbose:
            if i % print_every == 0:
                print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)
        loss_values.append(loss.item())

    return loss_values


def train_with_generated_data(ref_data, mix_object, deep_nmf, optimizerADAM):
    n_iter = 1
    train_size = 4000

    CTNames = ref_data[0]
    ref_data = ref_data[ref_data.abs().sum(dim=1).bool(), :]
    # ref_data = remove0s(ref_data)

    for train_index in range(train_size):
        v = generate_dists(ref_data.T, train_index * 0.0001)
        v_train_i = torch.from_numpy(v[0]).float().T
        v_train_i = v_train_i[v_train_i.abs().sum(dim=1).bool(), :]

        resultV = np.zeros((v_train_i.shape[0] + 1, v_train_i.shape[1] + 1), dtype="U28")
        resultV[0, 1:] = mix_object[0][1:]
        resultV[:, 0] = np.asanyarray(mix_object)[:, 0]
        resultV[1:, 1:] = v_train_i.detach().numpy()

        dist_train_i = torch.from_numpy(v[1]).float()

        normMix, normRef = qNormMatrices(resultV, ref_data)
        sharedMix, sharedRef = getSharedRows(normMix, ref_data)

        SigRef = returnSigMatrix([CTNames] + sharedRef, 50, 50 * len(ref_data[0]) - 1, "Entropy")
        SigMix, SigRef = getSharedRows(sharedMix, SigRef)
        ScaledRef, ScaledMix = RescaleRows(SigRef, SigMix, 0.0)
        _, Scaledv_train_i = ScaledRef, ScaledMix
        # _, Scaledv_train_i = run_gedit_pre1(tmp_file, ref_path, True)
        Scaledv_train_i = np.asanyarray(Scaledv_train_i).T[1:, 1:]
        Scaledv_train_i = torch.from_numpy(Scaledv_train_i.astype(float)).float()
        loss_values = train_supervised_one_sample(Scaledv_train_i, dist_train_i, n_iter, deep_nmf, optimizerADAM, False)
        if train_index % 1000 == 0:
            print(f"Train: Start train index: {train_index} with loss: {loss_values[-1]}")


def train_supervised_one_sample_reformat(
    v_train, h_train, n_components_ref, network_train_iterations, deep_nmf, optimizer, verbose=False, print_every=100
):
    n_h_rows, n_components = h_train.shape
    H_init_np = np.ones((n_h_rows, n_components_ref))
    H_init = torch.from_numpy(H_init_np).float()

    criterion = nn.MSELoss(reduction="mean")

    # Train the Network
    loss_values = []
    for i in range(network_train_iterations):
        out = deep_nmf(H_init, v_train)

        """
        out_result = np.zeros((out.shape[0] + 1, out.shape[1] + 1), dtype='U28')
        out_result[0, 1:] = cells_in_ref
        out_result[1:, 1:] = out.detach().numpy()
        out_result[1:, 0] = [f"Mixture_{i}" for i in range(len(out[:, 0]))]

        reformated_out = main_format(out_result, true_prop_path)
        reformated_out[0] = "\tmix" + reformated_out[0]

        reformated_out = torch.from_numpy(read_dataset_data(reformated_out)[1:, 1:].astype(float)).float()
        loss = torch.sqrt(criterion(reformated_out, h_train))  # loss between predicted and truth
        """
        loss = torch.sqrt(criterion(out, h_train))  # loss between predicted and truth
        if verbose:
            if i % print_every == 0:
                print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=math.inf)
        loss_values.append(loss.item())

    return loss_values
