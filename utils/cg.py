from functools import reduce
import numpy as np
import torch
from scipy.optimize import fmin_cg
from typing import Tuple
import torch.nn.functional as F



def my_cg(data, model, _edge_index, deltas, damping, device):
    inverse_hvp = []
    status = []
    cg_loss = []
    y_hat = model(data.x_unlearn, data.adj_t_unlearn)
    train_loss = F.cross_entropy(y_hat[data.retrain_mask], data.y[data.retrain_mask], reduction='sum')
    _, train_loss = _as_tuple(train_loss, "outputs of the user-provided function", "hvp")

    for i, (delta, param) in enumerate(zip(deltas, model.parameters())):
        param = [param]
        with torch.enable_grad():
            jac = _autograd_grad(train_loss, param, create_graph=True,retain_graph=True)
            grad_jac = tuple(
                torch.zeros_like(p, requires_grad=True, device=device) for p in param
            )
            double_back = _autograd_grad(jac, param, grad_jac, create_graph=True,retain_graph=True)
        delta = delta.detach()
        sizes = [delta.size()]
        fmin_loss_fn = get_fmin_loss_fn(delta, model=model, double_back=double_back, grad_jac=grad_jac,
                                        damping=damping, device=device, p_idx=i, sizes=sizes)
        fmin_grad_fn = get_fmin_grad_fn(delta, model=model, double_back=double_back, grad_jac=grad_jac,
                                        damping=damping, device=device, p_idx=i, sizes=sizes)

        res = fmin_cg(
            f=fmin_loss_fn,
            # x0=np.zeros_like(to_vector(v)),
            x0=to_vector(delta),
            fprime=fmin_grad_fn,
            gtol=1E-4,
            # norm='fro',
            # callback=cg_callback,
            disp=False,
            full_output=True,
            maxiter=30,
        )
        inverse_hvp.append(to_list(res[0], sizes, device)[0])
        # print('-----------------------------------')
        status.append(res[4])
        cg_loss.append(res[1])

    return inverse_hvp, np.mean(cg_loss), status


def get_fmin_loss_fn(delta, model, double_back, grad_jac, damping, device, p_idx, sizes):
    def _get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = get_hvp(x, model, double_back, grad_jac, damping, p_idx, sizes)
        # hvp_2 = get_vhp(x, model, double_back, jac, damping, p_idx, sizes)
        # print(torch.allclose(hvp, hvp_2))
        obj = 0.5 * torch.dot(hvp, x) - torch.dot(delta.view(-1), x)
        return obj.detach().cpu().numpy()

    return _get_fmin_loss


def get_fmin_grad_fn(delta, model, double_back, grad_jac, damping, device, p_idx, sizes):
    def _get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = get_hvp(x, model, double_back, grad_jac, damping, p_idx, sizes)
        return to_vector(hvp - delta.view(-1))

    return _get_fmin_grad


def to_vector(v):
    if isinstance(v, tuple) or isinstance(v, list):
        # return v.cpu().numpy().reshape(-1)
        return np.concatenate([vv.cpu().numpy().reshape(-1) for vv in v])
    else:
        return v.cpu().numpy().reshape(-1)


def to_list(v, sizes, device):
    _v = v
    result = []
    for size in sizes:
        total = reduce(lambda a, b: a * b, size)
        result.append(torch.from_numpy(_v[:total].reshape(size)).float().to(device))
        _v = _v[total:]
    return tuple(result)
    # return torch.tensor(v.reshape(sizes[0]), dtype=torch.float, device=device)


def get_hvp(x: torch.tensor, model, double_back, grad_jac, damping, p_idx, sizes):
    _hvp = hessian_vector_product(model, double_back, grad_jac, (x.view(sizes[0]),), p_idx)
    hvp = [b for b in _hvp]
    return hvp[0].view(-1) + damping * x



def hessian_vector_product(model, double_back, grad_jac, v, p_idx=None):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if p_idx is not None:
        parameters = parameters[p_idx:p_idx + 1]

    grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=True,retain_graph=True)
    hvp = _fill_in_zeros(grad_res, parameters, False, False, "double_back_trick")
    hvp = _grad_postprocess(hvp, False)
    return hvp


def _as_tuple(inp, arg_name=None, fn_name=None):
    # Ensures that inp is a tuple of Tensors
    # Returns whether or not the original inp was a tuple and the tupled version of the input
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)

    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False

    for i, el in enumerate(inp):
        if not isinstance(el, torch.Tensor):
            if is_inp_tuple:
                raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                " value at index {} has type {}.".format(arg_name, fn_name, i, type(el)))
            else:
                raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                " given {} has type {}.".format(arg_name, fn_name, arg_name, type(el)))

    return is_inp_tuple, inp


def _autograd_grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None):
    # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
    # This has the extra constraint that inputs has to be a tuple
    assert isinstance(outputs, tuple)
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    assert isinstance(grad_outputs, tuple)
    assert len(outputs) == len(grad_outputs)

    new_outputs: Tuple[torch.Tensor, ...] = tuple()
    new_grad_outputs: Tuple[torch.Tensor, ...] = tuple()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        # No differentiable output, we don't need to call the autograd engine
        return (None,) * len(inputs)
    else:
        return torch.autograd.grad(new_outputs, inputs, new_grad_outputs, allow_unused=True,
                                   create_graph=create_graph, retain_graph=retain_graph)


def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    # Used to detect None in the grads and depending on the flags, either replace them
    # with Tensors full of 0s of the appropriate size based on the refs or raise an error.
    # strict and create graph allow us to detect when it is appropriate to raise an error
    # stage gives us information of which backward call we consider to give good error message
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError("Invalid stage argument '{}' to _fill_in_zeros".format(stage))

    res = tuple()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == "back":
                    raise RuntimeError("The output of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode.".format(i))
                elif stage == "back_trick":
                    raise RuntimeError("The gradient with respect to the input is independent of entry {}"
                                       " in the grad_outputs when using the double backward trick to compute"
                                       " forward mode gradients. This is not allowed in strict mode.".format(i))
                elif stage == "double_back":
                    raise RuntimeError("The jacobian of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode.".format(i))
                else:
                    raise RuntimeError("The hessian of the user-provided function is independent of "
                                       "entry {} in the grad_jacobian. This is not allowed in strict "
                                       "mode as it prevents from using the double backward trick to "
                                       "replace forward mode AD.".format(i))

            grads_i = torch.zeros_like(refs[i])
        else:
            if strict and create_graph and not grads_i.requires_grad:
                if "double" not in stage:
                    raise RuntimeError("The jacobian of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
                else:
                    raise RuntimeError("The hessian of the user-provided function is independent of "
                                       "input {}. This is not allowed in strict mode when create_graph=True.".format(i))

        res += (grads_i,)

    return res


def _grad_postprocess(inputs, create_graph):
    # Postprocess the generated Tensors to avoid returning Tensors with history when the user did not
    # request it.
    if isinstance(inputs[0], torch.Tensor):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)


def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,
