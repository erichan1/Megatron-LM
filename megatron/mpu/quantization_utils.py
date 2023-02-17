import torch

QUANTIZATION_LEVEL = 0 # 0 == None, 1 == LLMint8, 2 == Smoothquant W8A16, 3 == Smoothquant W8A8
QUANTIZATION_IS_LOAD_STATE_DICT = True # Only flip to False for benchmarking purposes if not loading state dict

if QUANTIZATION_LEVEL == 0:
    pass # no quantization, do nothing
elif QUANTIZATION_LEVEL == 1:
    from bitsandbytes.nn import Linear8bitLt, Int8Params
elif QUANTIZATION_LEVEL == 2:
    from torch_int.nn.linear import W8A16Linear
elif QUANTIZATION_LEVEL == 3:
    raise Exception("Quantization level 3 currently not supported")
    from torch_int.nn.linear import W8A8B8O8Linear, W8A8BFP32OFP32Linear
    # Put in W8A8 stuff here eventually
else:
    raise Exception("This quantization level is not supported")

def create_llmint8_linear(weight, bias=None, has_fp16_weights=False, threshold=6.0, index=None):
    """
    weight: any kind of weight is fine. fp32, bf16, or fp16. We assume this is a CPU weight, to be converted to int8 upon sending to GPU.
        From Tim Dettmers: "when cuda() or to(device) is called, Int8Param class should intercept, cast the CPU weight to fp16 and do the transformation to int8 and then cast it to device/cuda."
    bias: the actual bias tensor. Can also be fp32, bf16, or f16 (optional)
    Other arg explanations TBD
    """
    output_features, input_features = weight.shape
    has_bias = bias is not None
    q_linear = Linear8bitLt(input_features, output_features, bias=has_bias, has_fp16_weights=has_fp16_weights, threshold=threshold, index=index)
    q_linear.weight = weight
    return q_linear

def quantized_inference_post_hook(module, incompatible_keys=None):
    """
    Holds configs on what linear to create.
    Decides on what linear to use, and sends it to cuda in the correct way so you don't blow up your GPU memory.
    """
    # we assume that the weight has been put on CPU and we disabled the initialization
    has_bias = module.bias is not None
    if has_bias:
        raise Exception("Int8 conversion currently does not support bias.")

    if QUANTIZATION_LEVEL == 1:
        module.weight = Int8Params(data=module.weight, has_fp16_weights=False, requires_grad=False) # on CPU
        # recommended threshold is 6.0, but can tweak. see llm_int8 paper for how to set
        module.q_linear = create_llmint8_linear(module.weight, bias=None, has_fp16_weights=False, threshold=6.0, index=None)
        module.q_linear.to(torch.cuda.current_device()) # send it over and get int8!
    elif QUANTIZATION_LEVEL == 2:
        output_features, input_features = module.weight.shape
        # create a temporary linear for W8A16Linear to latch onto that's empty
        temp_linear = torch.nn.Linear(input_features, output_features, bias=has_bias, device="meta")
        temp_linear.weight = module.weight
        temp_linear = temp_linear.cuda()
        module.q_linear = W8A16Linear.from_float(temp_linear)
        module.q_linear.dequant_type = module.dtype
        # clean up old weight
        del temp_linear
        module.weight = None
    else:
        raise Exception("Other quantization levels not currently supported")

def quantized_inference_pre_hook(module, state_dict=None, prefix=None, local_metadata=None, strict=None, missing_keys=None, unexpected_keys=None, error_msgs=None):
    """
    Create module weight right before state dict load so you don't blow up CPU memory.
    CPU weight created here will be immediately moved to GPU post load_state_dict on this particular weight, so won't hang around in CPU long.
    Unnecessary args are just to match the _register_load_state_dict_pre_hook method signature
    """
    module.weight = torch.nn.Parameter(torch.empty((module.quantized_output_size, module.quantized_input_size), requires_grad=False, dtype=module.dtype, device="cpu"))

def quantization_init(module, input_size, output_size, dtype):
    """
    If not quantizing, returns None. Otherwise, returns True.
    If quantizing, will replace module.q_linear with chosen int8 linear implementation when loading checkpoint into model.
    """
    if QUANTIZATION_LEVEL == 0:
        return None

    # be careful that these are not overriding some other parameter in the module.
    # necessary because hooks cannot take arguments besides module itself
    module.quantized_input_size = input_size
    module.quantized_output_size = output_size
    module.dtype = dtype

    if QUANTIZATION_IS_LOAD_STATE_DICT is True:
        module._register_load_state_dict_pre_hook(quantized_inference_pre_hook, with_module=True)
        module.register_load_state_dict_post_hook(quantized_inference_post_hook)
    else:
        # if we aren't loading state dict, call hooks directly to initialize quantized weights on model creation
        quantized_inference_pre_hook(module)
        quantized_inference_post_hook(module)
    return True # A placeholder to represent that module.q_linear will be replaced during load_state_dict on model
