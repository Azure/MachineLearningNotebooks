import numpy as np
import tritonhttpclient
from PIL import Image
from tritonclientutils import InferenceServerException, triton_to_np_dtype


def preprocess(img, scaling, dtype):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')
    c = 3
    h = 224
    w = 224
    format = "FORMAT_NCHW"
    
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == "FORMAT_NCHW":
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """

    output_array = results.as_numpy(output_name)
    if len(output_array) != batch_size:
        raise Exception("expected {} results, got {}".format(
            batch_size, len(output_array)))

    # Include special handling for non-batching models
    output = ""
    for results in output_array:
        if not batching:
            results = [results]
        for result in results:
            if output_array.dtype.type == np.bytes_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            output += "    {} ({}) = {}".format(cls[0], cls[1], cls[2])

    return output

def triton_init(url="localhost:8000"):
    global triton_client
    triton_client = tritonhttpclient.InferenceServerClient(url)
    _url = "localhost:8000"
    _model = "densenet_onnx"
    _scaling = "INCEPTION"

# TODO: set input_name, output_name, dtype from model metadata, as in
# https://github.com/triton-inference-server/server/blob/master/src/clients/python/examples/image_client.py#L402
def triton_infer(model_name, img, input_name="data_0", output_name="fc6_1",
                 dtype="FP32"):
    image_data = preprocess(img, scaling="INCEPTION", dtype=dtype)
    input = tritonhttpclient.InferInput(input_name, image_data.shape, dtype)
    input.set_data_from_numpy(image_data, binary_data=True)
    output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=True, class_count=1)

    res = triton_client.infer(model_name,
                              [input],
                              request_id="0",
                              outputs=[output])
    
    return postprocess(res, output_name=output_name, batch_size=1, batching=False)