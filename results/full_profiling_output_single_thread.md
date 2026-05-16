# Workload 2


```text
(cs348k-py3.12) jdarve@wheat-01:~/cs348k$ cat profiles/W2/*
=== InternVL2.5 HF Fast (W2) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  154.15 MB
peak / output:   2.00x

         6306 function calls (6273 primitive calls) in 0.473 seconds

   Ordered by: cumulative time
   List reduced from 168 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.005    0.003    0.468    0.468 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:99(<lambda>)
        1    0.000    0.000    0.468    0.468 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.468    0.468 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.052    0.052    0.418    0.418 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.007    0.007    0.365    0.365 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:781(_preprocess)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.127    0.127    0.127    0.127 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        3    0.118    0.039    0.118    0.039 {built-in method torch.stack}
        1    0.005    0.005    0.112    0.112 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
        2    0.000    0.000    0.064    0.032 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
        1    0.000    0.000    0.059    0.059 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
        1    0.000    0.000    0.059    0.059 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)
        1    0.000    0.000    0.059    0.059 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
        1    0.052    0.052    0.052    0.052 {method 'sub' of 'torch._C.TensorBase' objects}
        1    0.000    0.000    0.050    0.050 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.050    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
       32    0.000    0.000    0.049    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)


=== InternVL2.5 HF Legacy (W2) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  52.86 MB
peak / output:   0.69x

         12418 function calls (12354 primitive calls) in 0.598 seconds

   Ordered by: cumulative time
   List reduced from 116 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.598    0.598 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:93(<lambda>)
        1    0.004    0.004    0.598    0.598 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.000    0.000    0.594    0.594 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:202(preprocess)
       32    0.000    0.000    0.412    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:153(resize)
       32    0.000    0.000    0.412    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.000    0.000    0.337    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.336    0.010    0.336    0.010 {method 'resize' of 'ImagingCore' objects}
       32    0.000    0.000    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.074    0.002    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
       32    0.000    0.000    0.065    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
      128    0.018    0.000    0.057    0.000 {built-in method numpy.array}
       32    0.001    0.000    0.052    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
       32    0.000    0.000    0.052    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
       32    0.000    0.000    0.052    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
       32    0.000    0.000    0.047    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
       96    0.040    0.000    0.040    0.000 {method 'astype' of 'numpy.ndarray' objects}
       64    0.000    0.000    0.039    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.002    0.000    0.038    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.036    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.008    0.000    0.036    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)


=== InternVL2.5 Manual Card (W2) ===
output shape:    (320, 3, 448, 448)
output bytes:    770.70 MB
peak RSS delta:  1372.09 MB
peak / output:   1.78x

         69831 function calls (69830 primitive calls) in 4.557 seconds

   Ordered by: cumulative time
   List reduced from 125 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    1.588    0.794    1.531    1.531 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:105(<lambda>)
        1    0.001    0.001    1.490    1.490 /home/users/jdarve/cs348k/benchmarks/models.py:104(process_batch)
       32    0.003    0.000    1.307    0.041 /home/users/jdarve/cs348k/benchmarks/models.py:59(dynamic_preprocess)
       64    0.001    0.000    1.257    0.020 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       64    1.255    0.020    1.255    0.020 {method 'resize' of 'ImagingCore' objects}
        1    0.592    0.592    0.592    0.592 {built-in method torch.cat}
      320    0.002    0.000    0.575    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:93(__call__)
       32    0.494    0.015    0.494    0.015 {built-in method torch.stack}
      320    0.001    0.000    0.364    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:129(__call__)
      320    0.006    0.000    0.363    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:127(to_tensor)
      640    0.002    0.000    0.208    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1775(_wrapped_call_impl)
      640    0.002    0.000    0.206    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1783(_call_impl)
      320    0.000    0.000    0.193    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:277(forward)
      320    0.002    0.000    0.192    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:327(normalize)
      320    0.009    0.000    0.188    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py:905(normalize)
      320    0.021    0.000    0.098    0.000 {built-in method numpy.array}
      320    0.097    0.000    0.097    0.000 {method 'div' of 'torch._C.TensorBase' objects}
      320    0.094    0.000    0.094    0.000 {method 'contiguous' of 'torch._C.TensorBase' objects}
      320    0.088    0.000    0.088    0.000 {method 'clone' of 'torch._C.TensorBase' objects}
      320    0.001    0.000    0.077    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)


=== LLaVA fast (W2) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  507.91 MB
peak / output:   2.34x

         14253 function calls (14221 primitive calls) in 0.837 seconds

   Ordered by: cumulative time
   List reduced from 159 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.837    0.837 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:117(<lambda>)
        1    0.000    0.000    0.837    0.837 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.837    0.837 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:79(preprocess)
        1    0.000    0.000    0.837    0.837 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.018    0.018    0.837    0.837 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.015    0.015    0.766    0.766 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:203(_preprocess)
       65    0.287    0.004    0.287    0.004 {built-in method torch.stack}
       32    0.000    0.000    0.166    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
       32    0.000    0.000    0.155    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:137(_get_image_patches)
       32    0.000    0.000    0.146    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
       96    0.000    0.000    0.146    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       32    0.000    0.000    0.146    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)
       32    0.000    0.000    0.146    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
       96    0.002    0.000    0.145    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       64    0.000    0.000    0.144    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:5349(pad)
       64    0.144    0.002    0.144    0.002 {built-in method torch._C._nn.pad}
       64    0.001    0.000    0.140    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.000    0.000    0.139    0.139 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:181(_pad_for_batching)
       64    0.139    0.002    0.139    0.002 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       32    0.131    0.004    0.131    0.004 {method 'sub' of 'torch._C.TensorBase' objects}


=== LLaVA legacy (W2) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  649.42 MB
peak / output:   3.00x

         57839 function calls (57775 primitive calls) in 1.693 seconds

   Ordered by: cumulative time
   List reduced from 160 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.693    1.693 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:111(<lambda>)
        1    0.020    0.020    1.693    1.693 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.001    0.001    1.673    1.673 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:550(preprocess)
      224    0.002    0.000    1.038    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.000    0.000    0.985    0.031 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:447(get_image_patches)
      224    0.001    0.000    0.798    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       64    0.789    0.012    0.789    0.012 {method 'resize' of 'ImagingCore' objects}
       32    0.000    0.000    0.572    0.018 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:401(_resize_for_patching)
       32    0.001    0.000    0.395    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:317(_preprocess)
      160    0.000    0.000    0.230    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
      160    0.227    0.001    0.230    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
      832    0.119    0.000    0.191    0.000 {built-in method numpy.array}
      224    0.002    0.000    0.180    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
      224    0.003    0.000    0.144    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
      224    0.001    0.000    0.141    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
      224    0.001    0.000    0.140    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
       64    0.000    0.000    0.106    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:249(pad)
       64    0.001    0.000    0.105    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/lib/arraypad.py:533(pad)
       64    0.095    0.001    0.096    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/lib/arraypad.py:86(_pad_simple)
        1    0.000    0.000    0.096    0.096 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:511(_pad_for_batching)


=== Qwen2.5-VL fast (W2) ===
output shape:    (175232, 1176)
output bytes:    824.29 MB
peak RSS delta:  3090.86 MB
peak / output:   3.75x

         6259 function calls (6227 primitive calls) in 3.739 seconds

   Ordered by: cumulative time
   List reduced from 138 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    3.739    3.739 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:87(<lambda>)
        1    0.000    0.000    3.739    3.739 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    3.739    3.739 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:134(preprocess)
        1    0.000    0.000    3.739    3.739 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.172    0.172    3.739    3.739 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:143(_preprocess_image_like_inputs)
        1    0.033    0.033    3.516    3.516 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:182(_preprocess)
        2    1.193    0.597    1.193    0.597 {built-in method torch.cat}
        3    0.924    0.308    0.924    0.308 {method 'reshape' of 'torch._C.TensorBase' objects}
        1    0.026    0.026    0.621    0.621 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
        1    0.000    0.000    0.331    0.331 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
        1    0.000    0.000    0.331    0.331 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)
        1    0.000    0.000    0.331    0.331 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
        1    0.000    0.000    0.297    0.297 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.297    0.297 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.297    0.297 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.297    0.297 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.296    0.296    0.296    0.296 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        1    0.296    0.296    0.296    0.296 {method 'sub' of 'torch._C.TensorBase' objects}
        1    0.264    0.264    0.264    0.264 {method 'repeat' of 'torch._C.TensorBase' objects}
        1    0.264    0.264    0.264    0.264 {method 'to' of 'torch._C.TensorBase' objects}


=== Qwen2.5-VL legacy (W2) ===
output shape:    (175232, 1176)
output bytes:    824.29 MB
peak RSS delta:  1666.04 MB
peak / output:   2.02x

         14902 function calls (14838 primitive calls) in 2.604 seconds

   Ordered by: cumulative time
   List reduced from 117 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.604    2.604 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:81(<lambda>)
        1    0.009    0.009    2.604    2.604 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.008    0.008    2.595    2.595 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:299(preprocess)
       32    0.161    0.005    2.270    0.071 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:165(_preprocess)
       32    0.001    0.000    0.696    0.022 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.000    0.000    0.591    0.018 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.590    0.018    0.590    0.018 {method 'resize' of 'ImagingCore' objects}
      162    0.433    0.003    0.515    0.003 {built-in method numpy.array}
       64    0.475    0.007    0.475    0.007 {method 'reshape' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.449    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.448    0.014    0.449    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
       32    0.000    0.000    0.240    0.007 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.070    0.002    0.240    0.007 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       96    0.179    0.002    0.179    0.002 {method 'astype' of 'numpy.ndarray' objects}
       64    0.001    0.000    0.082    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.003    0.000    0.081    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:423(repeat)
       32    0.000    0.000    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
       32    0.075    0.002    0.075    0.002 {method 'repeat' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.063    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
```


# Workload 3

```text
(cs348k-py3.12) jdarve@wheat-01:~/cs348k/profiles/W3$ cat *
=== InternVL2.5 HF Fast (W3) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  0.00 MB
peak / output:   0.00x

         7281 function calls (7249 primitive calls) in 0.287 seconds

   Ordered by: cumulative time
   List reduced from 133 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.287    0.287 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:98(<lambda>)
        1    0.000    0.000    0.287    0.287 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.287    0.287 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.000    0.000    0.287    0.287 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.001    0.001    0.248    0.248 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:781(_preprocess)
       32    0.000    0.000    0.155    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
       32    0.000    0.000    0.154    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       32    0.001    0.000    0.153    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       32    0.001    0.000    0.150    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       32    0.149    0.005    0.149    0.005 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       62    0.062    0.001    0.062    0.001 {built-in method torch.stack}
        2    0.000    0.000    0.045    0.023 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
        1    0.000    0.000    0.038    0.038 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.038    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
       32    0.001    0.000    0.037    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
       32    0.008    0.000    0.034    0.001 {built-in method numpy.array}
       29    0.000    0.000    0.029    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
       32    0.000    0.000    0.026    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       32    0.001    0.000    0.026    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       29    0.000    0.000    0.021    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)


=== InternVL2.5 HF Legacy (W3) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  0.00 MB
peak / output:   0.00x

         10944 function calls (10880 primitive calls) in 0.716 seconds

   Ordered by: cumulative time
   List reduced from 116 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.716    0.716 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:92(<lambda>)
        1    0.000    0.000    0.716    0.716 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.001    0.001    0.716    0.716 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:202(preprocess)
       32    0.000    0.000    0.461    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:153(resize)
       32    0.001    0.000    0.460    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.001    0.000    0.397    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.395    0.012    0.395    0.012 {method 'resize' of 'ImagingCore' objects}
       32    0.000    0.000    0.135    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.134    0.004    0.135    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
      128    0.014    0.000    0.058    0.000 {built-in method numpy.array}
       32    0.000    0.000    0.054    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.017    0.001    0.054    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       96    0.046    0.000    0.046    0.000 {method 'astype' of 'numpy.ndarray' objects}
       64    0.001    0.000    0.044    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.002    0.000    0.043    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.001    0.000    0.038    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
       32    0.000    0.000    0.034    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
        1    0.000    0.000    0.028    0.028 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/feature_extraction_utils.py:77(__init__)
        1    0.000    0.000    0.028    0.028 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/feature_extraction_utils.py:171(convert_to_tensors)
        1    0.000    0.000    0.028    0.028 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/feature_extraction_utils.py:160(as_tensor)


=== InternVL2.5 Manual Card (W3) ===
output shape:    (253, 3, 448, 448)
output bytes:    609.34 MB
peak RSS delta:  958.05 MB
peak / output:   1.57x

         57570 function calls (57569 primitive calls) in 3.376 seconds

   Ordered by: cumulative time
   List reduced from 125 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.114    0.057    3.299    3.299 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:104(<lambda>)
        1    0.002    0.002    2.181    2.181 /home/users/jdarve/cs348k/benchmarks/models.py:104(process_batch)
       32    0.004    0.000    1.472    0.046 /home/users/jdarve/cs348k/benchmarks/models.py:59(dynamic_preprocess)
       62    0.001    0.000    1.411    0.023 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       62    1.409    0.023    1.409    0.023 {method 'resize' of 'ImagingCore' objects}
      253    0.005    0.000    0.694    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:93(__call__)
        1    0.677    0.677    0.677    0.677 {built-in method torch.cat}
      253    0.001    0.000    0.449    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:129(__call__)
      253    0.009    0.000    0.449    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:127(to_tensor)
       32    0.417    0.013    0.417    0.013 {built-in method torch.stack}
      506    0.002    0.000    0.239    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1775(_wrapped_call_impl)
      506    0.003    0.000    0.236    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1783(_call_impl)
      253    0.001    0.000    0.216    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:277(forward)
      253    0.003    0.000    0.215    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:327(normalize)
      253    0.013    0.000    0.210    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py:905(normalize)
      253    0.129    0.001    0.129    0.001 {method 'contiguous' of 'torch._C.TensorBase' objects}
      253    0.025    0.000    0.126    0.000 {built-in method numpy.array}
      253    0.106    0.000    0.106    0.000 {method 'div' of 'torch._C.TensorBase' objects}
      253    0.002    0.000    0.101    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
      253    0.009    0.000    0.097    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)


=== LLaVA fast (W3) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  385.82 MB
peak / output:   1.78x

         12463 function calls (12431 primitive calls) in 0.752 seconds

   Ordered by: cumulative time
   List reduced from 159 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.752    0.752 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:116(<lambda>)
        1    0.000    0.000    0.752    0.752 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.752    0.752 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:79(preprocess)
        1    0.000    0.000    0.752    0.752 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.000    0.000    0.751    0.751 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.015    0.015    0.714    0.714 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:203(_preprocess)
       65    0.287    0.004    0.287    0.004 {built-in method torch.stack}
       32    0.001    0.000    0.178    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:137(_get_image_patches)
       64    0.000    0.000    0.169    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:5349(pad)
       64    0.168    0.003    0.168    0.003 {built-in method torch._C._nn.pad}
       96    0.001    0.000    0.142    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.140    0.140 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:181(_pad_for_batching)
       96    0.003    0.000    0.140    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       63    0.001    0.000    0.132    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       63    0.130    0.002    0.130    0.002 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       32    0.001    0.000    0.087    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
       64    0.001    0.000    0.085    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
       32    0.000    0.000    0.084    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:83(_resize_for_patching)
       32    0.000    0.000    0.059    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
       32    0.000    0.000    0.059    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)


=== LLaVA legacy (W3) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  352.11 MB
peak / output:   1.62x

         52872 function calls (52808 primitive calls) in 1.705 seconds

   Ordered by: cumulative time
   List reduced from 160 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.704    1.704 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:110(<lambda>)
        1    0.005    0.005    1.704    1.704 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.001    0.001    1.699    1.699 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:550(preprocess)
      211    0.004    0.000    0.925    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.001    0.000    0.835    0.026 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:447(get_image_patches)
      211    0.002    0.000    0.717    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       63    0.701    0.011    0.701    0.011 {method 'resize' of 'ImagingCore' objects}
       32    0.002    0.000    0.593    0.019 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:317(_preprocess)
       32    0.000    0.000    0.472    0.015 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:401(_resize_for_patching)
      147    0.001    0.000    0.348    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
      147    0.343    0.002    0.347    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
      793    0.095    0.000    0.181    0.000 {built-in method numpy.array}
      211    0.003    0.000    0.121    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
      147    0.001    0.000    0.120    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
      147    0.039    0.000    0.119    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       64    0.001    0.000    0.115    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:249(pad)
       64    0.002    0.000    0.113    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/lib/arraypad.py:533(pad)
      147    0.001    0.000    0.112    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:199(resize)
      569    0.107    0.000    0.107    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.099    0.099 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:511(_pad_for_batching)


=== Qwen2.5-VL fast (W3) ===
output shape:    (76800, 1176)
output bytes:    361.27 MB
peak RSS delta:  557.72 MB
peak / output:   1.54x

         7552 function calls (7520 primitive calls) in 1.268 seconds

   Ordered by: cumulative time
   List reduced from 138 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.268    1.268 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:86(<lambda>)
        1    0.000    0.000    1.268    1.268 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    1.268    1.268 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:134(preprocess)
        1    0.000    0.000    1.268    1.268 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.022    0.022    1.268    1.268 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:143(_preprocess_image_like_inputs)
        1    0.004    0.004    1.207    1.207 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:182(_preprocess)
       33    0.527    0.016    0.527    0.016 {built-in method torch.cat}
       96    0.274    0.003    0.274    0.003 {method 'reshape' of 'torch._C.TensorBase' objects}
       32    0.000    0.000    0.191    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
       32    0.000    0.000    0.190    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       32    0.001    0.000    0.190    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       32    0.001    0.000    0.186    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       32    0.185    0.006    0.185    0.006 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       32    0.001    0.000    0.120    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
       32    0.000    0.000    0.082    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
       32    0.001    0.000    0.082    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)
       32    0.001    0.000    0.081    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
       32    0.061    0.002    0.061    0.002 {method 'sub' of 'torch._C.TensorBase' objects}
        2    0.000    0.000    0.049    0.024 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
       64    0.048    0.001    0.048    0.001 {built-in method torch.stack}


=== Qwen2.5-VL legacy (W3) ===
output shape:    (76800, 1176)
output bytes:    361.27 MB
peak RSS delta:  558.67 MB
peak / output:   1.55x

         11487 function calls (11422 primitive calls) in 1.782 seconds

   Ordered by: cumulative time
   List reduced from 152 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.002    0.001    1.725    1.725 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:80(<lambda>)
        1    0.066    0.066    1.725    1.725 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.006    0.006    1.658    1.658 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:299(preprocess)
       32    0.082    0.003    1.471    0.046 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:165(_preprocess)
       32    0.001    0.000    0.549    0.017 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.001    0.000    0.477    0.015 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.476    0.015    0.476    0.015 {method 'resize' of 'ImagingCore' objects}
       32    0.000    0.000    0.333    0.010 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.331    0.010    0.332    0.010 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
      162    0.251    0.002    0.309    0.002 {built-in method numpy.array}
       64    0.266    0.004    0.266    0.004 {method 'reshape' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.144    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.040    0.001    0.144    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       96    0.109    0.001    0.109    0.001 {method 'astype' of 'numpy.ndarray' objects}
       64    0.001    0.000    0.058    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.003    0.000    0.056    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.040    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:423(repeat)
       32    0.000    0.000    0.040    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
       32    0.039    0.001    0.039    0.001 {method 'repeat' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.039    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
```


# Workload 4

```text
(cs348k-py3.12) jdarve@wheat-01:~/cs348k/profiles/W4$ cat *
=== InternVL2.5 HF Fast (W4) ===
output shape:    (8, 3, 448, 448)
output bytes:    19.27 MB
peak RSS delta:  243.60 MB
peak / output:   12.64x

         8034 function calls (8025 primitive calls) in 1.197 seconds

   Ordered by: cumulative time
   List reduced from 168 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.000    0.000    1.197    1.197 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:98(<lambda>)
        1    0.000    0.000    1.197    1.197 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    1.196    1.196 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.000    0.000    1.026    1.026 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.021    0.021    0.664    0.664 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:781(_preprocess)
        1    0.000    0.000    0.629    0.629 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.429    0.429    0.629    0.629 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        3    0.368    0.123    0.368    0.123 {built-in method torch.stack}
        2    0.000    0.000    0.363    0.182 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
        1    0.000    0.000    0.199    0.199 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.199    0.199 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.199    0.199    0.199    0.199 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        1    0.000    0.000    0.170    0.170 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
        8    0.000    0.000    0.170    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
        8    0.000    0.000    0.170    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
        8    0.044    0.006    0.168    0.021 {built-in method numpy.array}
        8    0.000    0.000    0.124    0.016 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
        8    0.003    0.000    0.123    0.015 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
     3512    0.072    0.000    0.072    0.000 {method 'encode' of 'ImagingEncoder' objects}
        8    0.047    0.006    0.047    0.006 {method 'join' of 'bytes' objects}


=== InternVL2.5 HF Legacy (W4) ===
output shape:    (8, 3, 448, 448)
output bytes:    19.27 MB
peak RSS delta:  0.00 MB
peak / output:   0.00x

         9458 function calls (9442 primitive calls) in 1.374 seconds

   Ordered by: cumulative time
   List reduced from 116 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.374    1.374 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:92(<lambda>)
        1    0.000    0.000    1.374    1.374 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.000    0.000    1.374    1.374 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:202(preprocess)
        8    0.000    0.000    1.149    0.144 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:153(resize)
        8    0.000    0.000    1.149    0.144 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
        8    0.000    0.000    0.989    0.124 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
        8    0.989    0.124    0.989    0.124 {method 'resize' of 'ImagingCore' objects}
       32    0.046    0.001    0.175    0.005 {built-in method numpy.array}
        8    0.000    0.000    0.169    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
        8    0.000    0.000    0.153    0.019 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
       16    0.001    0.000    0.129    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       16    0.003    0.000    0.128    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
        8    0.000    0.000    0.108    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
        8    0.000    0.000    0.107    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
        8    0.000    0.000    0.107    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
     3624    0.075    0.000    0.075    0.000 {method 'encode' of 'ImagingEncoder' objects}
        8    0.000    0.000    0.064    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:932(frombytes)
        8    0.064    0.008    0.064    0.008 {method 'decode' of 'ImagingDecoder' objects}
       24    0.054    0.002    0.054    0.002 {method 'astype' of 'numpy.ndarray' objects}
       16    0.048    0.003    0.048    0.003 {method 'join' of 'bytes' objects}


=== InternVL2.5 Manual Card (W4) ===
output shape:    (56, 3, 448, 448)
output bytes:    134.87 MB
peak RSS delta:  134.88 MB
peak / output:   1.00x

         13052 function calls in 2.497 seconds

   Ordered by: cumulative time
   List reduced from 90 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.497    2.497 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:104(<lambda>)
        1    0.001    0.001    2.497    2.497 /home/users/jdarve/cs348k/benchmarks/models.py:104(process_batch)
        8    0.001    0.000    2.163    0.270 /home/users/jdarve/cs348k/benchmarks/models.py:59(dynamic_preprocess)
       16    0.000    0.000    2.150    0.134 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       16    2.149    0.134    2.149    0.134 {method 'resize' of 'ImagingCore' objects}
       56    0.001    0.000    0.152    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:93(__call__)
        1    0.152    0.152    0.152    0.152 {built-in method torch.cat}
       56    0.000    0.000    0.100    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:129(__call__)
       56    0.002    0.000    0.100    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:127(to_tensor)
      112    0.001    0.000    0.052    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1775(_wrapped_call_impl)
      112    0.001    0.000    0.051    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1783(_call_impl)
       56    0.000    0.000    0.047    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:277(forward)
       56    0.001    0.000    0.047    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:327(normalize)
       56    0.003    0.000    0.045    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py:905(normalize)
       56    0.029    0.001    0.029    0.001 {method 'contiguous' of 'torch._C.TensorBase' objects}
        8    0.028    0.004    0.028    0.004 {built-in method torch.stack}
       56    0.006    0.000    0.028    0.001 {built-in method numpy.array}
       56    0.023    0.000    0.023    0.000 {method 'div' of 'torch._C.TensorBase' objects}
       56    0.000    0.000    0.023    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       56    0.002    0.000    0.022    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)


=== LLaVA fast (W4) ===
output shape:    (8, 5, 3, 336, 336)
output bytes:    54.19 MB
peak RSS delta:  54.20 MB
peak / output:   1.00x

         9909 function calls (9901 primitive calls) in 0.570 seconds

   Ordered by: cumulative time
   List reduced from 159 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.570    0.570 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:116(<lambda>)
        1    0.000    0.000    0.570    0.570 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.570    0.570 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:79(preprocess)
        1    0.000    0.000    0.570    0.570 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.000    0.000    0.570    0.570 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.001    0.001    0.399    0.399 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:203(_preprocess)
        8    0.000    0.000    0.290    0.036 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:137(_get_image_patches)
       24    0.000    0.000    0.279    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       24    0.001    0.000    0.279    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       16    0.000    0.000    0.276    0.017 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       16    0.275    0.017    0.275    0.017 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        1    0.000    0.000    0.170    0.170 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
        8    0.000    0.000    0.170    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
        8    0.000    0.000    0.170    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
        8    0.045    0.006    0.168    0.021 {built-in method numpy.array}
       16    0.000    0.000    0.150    0.009 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        8    0.000    0.000    0.150    0.019 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:83(_resize_for_patching)
        8    0.001    0.000    0.124    0.015 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
        8    0.003    0.000    0.123    0.015 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       17    0.073    0.004    0.073    0.004 {built-in method torch.stack}


=== LLaVA legacy (W4) ===
output shape:    (8, 5, 3, 336, 336)
output bytes:    54.19 MB
peak RSS delta:  54.20 MB
peak / output:   1.00x

         20730 function calls (20713 primitive calls) in 2.509 seconds

   Ordered by: cumulative time
   List reduced from 195 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.001    0.000    2.508    2.508 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:110(<lambda>)
        1    0.000    0.000    2.508    2.508 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
       56    0.001    0.000    2.090    0.037 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
        8    0.039    0.005    1.986    0.248 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:447(get_image_patches)
       56    0.001    0.000    1.764    0.031 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       16    1.759    0.110    1.759    0.110 {method 'resize' of 'ImagingCore' objects}
        8    0.000    0.000    1.102    0.138 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:401(_resize_for_patching)
        1    0.000    0.000    0.322    0.322 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:550(preprocess)
       56    0.001    0.000    0.321    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
       56    0.002    0.000    0.226    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
       56    0.000    0.000    0.225    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
       56    0.001    0.000    0.224    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
      208    0.070    0.000    0.210    0.001 {built-in method numpy.array}
        8    0.000    0.000    0.168    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
        8    0.001    0.000    0.164    0.020 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:317(_preprocess)
       64    0.001    0.000    0.140    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.005    0.000    0.138    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       56    0.001    0.000    0.134    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:932(frombytes)
       56    0.132    0.002    0.132    0.002 {method 'decode' of 'ImagingDecoder' objects}
      152    0.115    0.001    0.115    0.001 {method 'astype' of 'numpy.ndarray' objects}


=== Qwen2.5-VL fast (W4) ===
output shape:    (356000, 1176)
output bytes:    1674.62 MB
peak RSS delta:  6279.77 MB
peak / output:   3.75x

         8054 function calls (8045 primitive calls) in 15.439 seconds

   Ordered by: cumulative time
   List reduced from 173 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.188    0.094   11.540   11.540 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:86(<lambda>)
        1    0.000    0.000   11.540   11.540 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000   11.540   11.540 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:134(preprocess)
        1    3.911    3.911   11.540   11.540 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.518    0.518    7.295    7.295 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:143(_preprocess_image_like_inputs)
        1    0.089    0.089    6.777    6.777 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:182(_preprocess)
        2    3.477    1.739    3.477    1.739 {built-in method torch.cat}
        3    2.603    0.868    2.603    0.868 {method 'reshape' of 'torch._C.TensorBase' objects}
        1    0.095    0.095    1.755    1.755 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
        1    0.000    0.000    1.359    1.359 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    1.359    1.359 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    1.359    1.359 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    1.359    1.359 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    1.359    1.359    1.359    1.359 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        1    0.000    0.000    0.889    0.889 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
        1    0.000    0.000    0.889    0.889 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)
        1    0.000    0.000    0.889    0.889 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
        1    0.802    0.802    0.802    0.802 {method 'sub' of 'torch._C.TensorBase' objects}
        1    0.772    0.772    0.772    0.772 {method 'to' of 'torch._C.TensorBase' objects}
        1    0.607    0.607    0.607    0.607 {method 'repeat' of 'torch._C.TensorBase' objects}


=== Qwen2.5-VL legacy (W4) ===
output shape:    (356000, 1176)
output bytes:    1674.62 MB
peak RSS delta:  3382.55 MB
peak / output:   2.02x

         16281 function calls (16264 primitive calls) in 11.541 seconds

   Ordered by: cumulative time
   List reduced from 152 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.011    0.005   10.774   10.774 /home/users/jdarve/cs348k/benchmarks/full_benchmark_single_thread.py:80(<lambda>)
        1    0.859    0.859   10.774   10.774 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.057    0.057    7.812    7.812 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:299(preprocess)
        8    0.571    0.071    7.484    0.935 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:165(_preprocess)
       42    2.722    0.065    2.976    0.071 {built-in method numpy.array}
        8    0.000    0.000    2.543    0.318 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
        8    0.000    0.000    2.215    0.277 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
        8    2.215    0.277    2.215    0.277 {method 'resize' of 'ImagingCore' objects}
        8    0.000    0.000    1.631    0.204 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
        8    1.631    0.204    1.631    0.204 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
       16    1.621    0.101    1.621    0.101 {method 'reshape' of 'numpy.ndarray' objects}
        8    0.000    0.000    1.026    0.128 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
        8    0.192    0.024    1.026    0.128 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       24    0.877    0.037    0.877    0.037 {method 'astype' of 'numpy.ndarray' objects}
        8    0.000    0.000    0.333    0.042 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:423(repeat)
        8    0.000    0.000    0.333    0.042 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
        8    0.332    0.042    0.332    0.042 {method 'repeat' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.253    0.016 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       16    0.007    0.000    0.252    0.016 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
        8    0.000    0.000    0.173    0.022 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
```
