# auto_norm

`auto_norm.build_norm_map` is the key entrypoint, it returns a `norm_map` function that computes computes (norms of inputs, norms of parameters, norms of buffers) -> norms of outputs.

Its syntax is

```py
def build_norm_map(module: nn.Module, *example_args, dynamic_shapes: Optional = None, **example_kwargs):
    ...

    def norm_map(*normed_args, normed_state_dict, **normed_kwargs):
        # normed_* should generally contain auto_norm.*_NormTensor, instead of usual torch.Tensor
        ...
        return normed_outputs

    return norm_map
```

The resulting `norm_map` can be backproped through with `torch.autograd`. Therefore, one can
1. Compute the Modula norm using the sensitivity definition
2. Optimize norm sizes of weight tensors and scalar scaling factors.

See [241105_auto_norm_ex.ipynb](./241105_auto_norm_ex.ipynb) for some examples.


## FAQ

**What is in a `NormedTensorBase`?**

It is a has-a subclass of `torch.Tensor`. In fact, it is a "has-two-tensors".
1. A "backing" tensor that this normed tensor is describing. It may not be a real tensor with actual data, but often, is a `FakeTensor` that has only metadata.

   In user land, when we create a normed tensor, we don't need to provide this backing tensor, as it will be automatically attached using information gathered from the module computation graph (specifically, the `FakeTensor`s created during `torch.export` in the **fake** mode, see below).
2. A `norm_size` tensor that could require gradient. Note that even if `normed_tensor.norm_size.requires_grad`, `normed_tensor.requires_grad=False` since  we are circumventing the autograd dispatch.

In addition, it also has
+ `elem_dims` that specifies which dimensions we are norm'ing over.

---
**How do I implement a normed rule for a PyTorch op or a custom function (composed of several ops)?**

Rather simple, for a PyTorch op, one can simply
```py
@auto_norm.reg_fake_norm_op(torch.ops.aten.relu.default).register_norm
def relu(input: RMS_NormTensor) -> RMS_NormTensor:
    return RMS_NormTensor(input.norm_size / np.sqrt(2), elem_dims=input.elem_dims)

# can work on factory functions too
@reg_fake_norm_op(torch.ops.aten.randn.default).register_norm
def randn(size: List[int], *, dtype: Optional[torch.dtype] = None, layout: Optional[torch.layout] = torch.strided, device: Optional[torch.device] = None, pin_memory: Optional[bool] = False) -> RMS_NormTensor:
    return RMS_NormTensor(1, elem_dims=None)

```

The `register_norm` function registers this normed implementation, with dispatch key as the **types** of Tensor arguments. Here, it is `(RMS_NormTensor,)`. When dispatching, we pick the first implementation where all argument matches in a `issubclass` relation.

For a custom op, one can provide defintion for both real and normed tensors as
```py
@reg_fake_norm_op
def mul_with_scaler(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert scale.ndim == 0
    return input * scale  # or your complicated op

@_mul_with_scaler.register_norm(allow_non_normed_tensor_inputs=True)  # allow_non_normed_tensor_inputs allows mixing normed and rebgular tensors for normed mode
def _(input: NormedTensorBase, scale: torch.Tensor) -> NormedTensorBase:
    assert scale.ndim == 0
    return input.__class__(input.norm_size * scale, elem_dims=input.elem_dims)


y = mul_with_scaler(x, scale)  # use this op this way
```

---
**How does `auto_norm` run normed tensors over real computation graphs?**

Under the hood, everything is implemented by PyTorch custom ops. An `auto_norm` wrapper custom op can wrap either (1) a PyTorch function (e.g., `torch.nn.functional.linear`) or (2) a custom function that can perform multiple operations (e.g., `zeropower_via_newtonschulz5`).

For these custom ops, we perform dispatching in three ways:

1. **regular** mode on real tensor (e.g., `torch.Tensor`)

   E.g., when running PyTorch eager.

2. **fake** mode on `FakeTensor`

   Used when we use PyTorch 2 `torch.export` to trace the model into a computation `torch.fx.Graph`. When exporting, there are generally two cases:
     1. The called function wraps a PyTorch function (e.g., the `torch.ops.aten.relu.default` example above).

        We use a `TorchFunctionMode` to see if any of the called op has an `auto_norm` wrapper registered, and instead dispatch to the wrapper, which knows how to handle `FakeTensor`.

    2. The called function wraps a custom function (e.g., the `mul_with_scaler` example above).

        In this case, the custom op is directly invoked. Hence we don't need extra dispatching.

3. **normed** mode on `NormedTensorBase` subclasses

   After tracing with fake mode, we obtain a `torch.fx.Graph` with **only** `auto_norm` wrapper custom ops. Then, we use another `TorchDispatchMode` to
    1. Call the normed implementations for the wrapper custom ops (which could call normal ops on regular `Tensor`s).
    2. Allow regular `Tensor` ops be invoked during normed implementation calls.

---
**How does `auto_norm` work?**

When given a module, we first `torch.export` it with the **fake** mode, converting it to a `torch.fx.Graph` with only wrapped custom ops. Then, `norm_map` run normed tensors on this exported graph with dispatch rules specified above in a **normed** mode.

---
## State of `auto_norm`

- [x] proof of concept that can trace general PyTorch modules and backprop through `norm_map`
- normed tensor op rules
    - [ ] figure out a nice API to perform `elem_dims` comparison and computation
    - [ ] fix some normed tensor rules to test `elem_dims`
    - [ ] implement more normed tensor rules so we can easily trace a GPT
- more functionals
    - [ ] initialization function `(norms) -> tensors`, supporting both real tensor mode and normed tensor mode, and is differentiable in the latter
    - [ ] a modular norm function `(norms, masses) -> modula coeffs`, differentiable
    - [ ] dualize
- UX
    - [ ] ways to annotate norm types in `nn.Module` definition
- norm type inference
    - [ ] infer norms based on dispatch implementations
- misc
    - [ ] optimize norm size to  $\propto$ modula maximal update step size?