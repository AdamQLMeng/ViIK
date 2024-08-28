import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch

from model.layers import mlp


def build_glow_model(dim_cond: tuple, dim_tot: int, num_nodes, x_invSig=None, x_Mu=None, coeff_fn_internal_size=1024, coeff_fn_layers=4, rnvp_clamp=2.5, is_cond=True, is_alternate=True):
    """
    Build an conditional invertible neural network consisting of a sequence of glow coupling layers, and permutation layers
    """

    def subnet_constructor_wrapper_1(ch_in: int, ch_out: int):
        return mlp(coeff_fn_internal_size, coeff_fn_layers, ch_in, ch_out)

    def subnet_constructor_wrapper_2(ch_in: int, ch_out: int):
        return mlp(coeff_fn_internal_size, coeff_fn_layers, ch_in, ch_out)

    subnet_constructor_wrapper = [subnet_constructor_wrapper_1, subnet_constructor_wrapper_2]

    # Input Node
    input_node = Ff.InputNode(dim_tot, name="input")
    nodes = [input_node]
    cond = []
    for dim_c in dim_cond:
        cond.append(Ff.ConditionNode(dim_c))

    # Clamp
    if x_invSig is not None and x_Mu is not None:
        nodes.append(Ff.Node([nodes[-1].out0], Fm.FixedLinearTransform, {"M": x_invSig, "b": x_Mu}))

    coupling_block = Fm.GLOWCouplingBlock

    split_dimension = dim_tot // 2  # // is a floor division operator

    for i in range(num_nodes):
        permute_node = Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {"seed": i})
        nodes.append(permute_node)
        if is_alternate:
            index = i % len(cond)
        else:
            index = i // (num_nodes//len(cond))
        cond_node = cond[index]
        subnet_constructor = subnet_constructor_wrapper[index]
        print("cond node: ", cond_node)
        glow_node = Ff.Node(
            nodes[-1].out0,
            coupling_block,
            {
                "subnet_constructor": subnet_constructor,
                "clamp": rnvp_clamp,
                # "clamp_activation": torch.nn.functional.gelu,
                "split_len": split_dimension,
            },
            conditions=cond_node if is_cond else None,  #  i % len(cond)
        )
        nodes.append(glow_node)

    model = Ff.GraphINN(nodes + cond + [Ff.OutputNode([nodes[-1].out0], name="output")], verbose=False)
    print(f"GLOW model: {model.node_list}")
    return model
