# Workload 2

```text
(cs348k-py3.12) jdarve@wheat-01:~/cs348k$ cat profiles/W2/*
=== InternVL2.5 HF Fast (W2) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  154.15 MB
peak / output:   2.00x

         6239 function calls (6207 primitive calls) in 0.274 seconds

   Ordered by: cumulative time
   List reduced from 133 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.273    0.273 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:99(<lambda>)
        1    0.000    0.000    0.273    0.273 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.273    0.273 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.006    0.006    0.273    0.273 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.009    0.009    0.218    0.218 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:781(_preprocess)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.127    0.127 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.127    0.127    0.127    0.127 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        1    0.000    0.000    0.049    0.049 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.049    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
        3    0.049    0.016    0.049    0.016 {built-in method torch.stack}
       32    0.000    0.000    0.048    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
       32    0.016    0.000    0.047    0.001 {built-in method numpy.array}
        2    0.000    0.000    0.037    0.019 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
        1    0.006    0.006    0.033    0.033 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
       32    0.000    0.000    0.031    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       32    0.001    0.000    0.031    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
     1568    0.021    0.000    0.021    0.000 {method 'encode' of 'ImagingEncoder' objects}


=== InternVL2.5 HF Legacy (W2) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  52.86 MB
peak / output:   0.69x

         12418 function calls (12354 primitive calls) in 0.590 seconds

   Ordered by: cumulative time
   List reduced from 116 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.589    0.589 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:93(<lambda>)
        1    0.003    0.003    0.589    0.589 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.000    0.000    0.586    0.586 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:202(preprocess)
       32    0.000    0.000    0.408    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:153(resize)
       32    0.000    0.000    0.408    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.000    0.000    0.337    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.336    0.011    0.336    0.011 {method 'resize' of 'ImagingCore' objects}
       32    0.000    0.000    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.074    0.002    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
       32    0.000    0.000    0.061    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
      128    0.018    0.000    0.055    0.000 {built-in method numpy.array}
       32    0.001    0.000    0.050    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
       32    0.000    0.000    0.049    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
       32    0.000    0.000    0.049    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
       32    0.000    0.000    0.047    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
       64    0.000    0.000    0.037    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       96    0.037    0.000    0.037    0.000 {method 'astype' of 'numpy.ndarray' objects}
       64    0.002    0.000    0.037    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.034    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.008    0.000    0.034    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)


=== InternVL2.5 Manual Card (W2) ===
output shape:    (320, 3, 448, 448)
output bytes:    770.70 MB
peak RSS delta:  1492.96 MB
peak / output:   1.94x

         69764 function calls in 3.286 seconds

   Ordered by: cumulative time
   List reduced from 90 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.075    0.075    3.286    3.286 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:105(<lambda>)
        1    0.003    0.003    3.211    3.211 /home/users/jdarve/cs348k/benchmarks/models.py:104(process_batch)
       32    0.005    0.000    2.395    0.075 /home/users/jdarve/cs348k/benchmarks/models.py:59(dynamic_preprocess)
       64    0.001    0.000    2.324    0.036 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       64    2.321    0.036    2.321    0.036 {method 'resize' of 'ImagingCore' objects}
      320    0.004    0.000    0.503    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:93(__call__)
      320    0.001    0.000    0.344    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:129(__call__)
      320    0.011    0.000    0.343    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:127(to_tensor)
       32    0.174    0.005    0.174    0.005 {built-in method torch.stack}
      320    0.039    0.000    0.174    0.001 {built-in method numpy.array}
      640    0.003    0.000    0.154    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1775(_wrapped_call_impl)
      640    0.004    0.000    0.151    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1783(_call_impl)
        1    0.136    0.136    0.136    0.136 {built-in method torch.cat}
      320    0.002    0.000    0.134    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
      320    0.010    0.000    0.129    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
      320    0.001    0.000    0.127    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:277(forward)
      320    0.003    0.000    0.126    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:327(normalize)
      320    0.016    0.000    0.119    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py:905(normalize)
     3200    0.086    0.000    0.086    0.000 {method 'encode' of 'ImagingEncoder' objects}
      320    0.075    0.000    0.075    0.000 {method 'contiguous' of 'torch._C.TensorBase' objects}


=== LLaVA fast (W2) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  533.88 MB
peak / output:   2.46x

         14253 function calls (14221 primitive calls) in 0.647 seconds

   Ordered by: cumulative time
   List reduced from 159 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.647    0.647 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:117(<lambda>)
        1    0.000    0.000    0.647    0.647 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.647    0.647 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:79(preprocess)
        1    0.000    0.000    0.647    0.647 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.013    0.013    0.647    0.647 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.031    0.031    0.563    0.563 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:203(_preprocess)
       32    0.001    0.000    0.358    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:137(_get_image_patches)
       96    0.001    0.000    0.346    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       96    0.004    0.000    0.344    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       64    0.001    0.000    0.334    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       64    0.332    0.005    0.332    0.005 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       64    0.001    0.000    0.211    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
       32    0.000    0.000    0.210    0.007 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:83(_resize_for_patching)
       65    0.080    0.001    0.080    0.001 {built-in method torch.stack}
        1    0.000    0.000    0.070    0.070 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.070    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
       32    0.000    0.000    0.069    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
       32    0.019    0.001    0.067    0.002 {built-in method numpy.array}
       64    0.000    0.000    0.051    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:5349(pad)
       64    0.051    0.001    0.051    0.001 {built-in method torch._C._nn.pad}


=== LLaVA legacy (W2) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  654.23 MB
peak / output:   3.02x

         57906 function calls (57841 primitive calls) in 2.894 seconds

   Ordered by: cumulative time
   List reduced from 195 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.001    0.001    2.892    2.892 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:111(<lambda>)
        1    0.035    0.035    2.892    2.892 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
      224    0.005    0.000    1.817    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.001    0.000    1.718    0.054 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:447(get_image_patches)
      224    0.003    0.000    1.395    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       64    1.376    0.022    1.376    0.022 {method 'resize' of 'ImagingCore' objects}
       32    0.003    0.000    0.921    0.029 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:401(_resize_for_patching)
       32    0.003    0.000    0.724    0.023 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:317(_preprocess)
      160    0.001    0.000    0.443    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
      160    0.437    0.003    0.442    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
        1    0.000    0.000    0.365    0.365 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:550(preprocess)
      224    0.003    0.000    0.322    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
      832    0.183    0.000    0.305    0.000 {built-in method numpy.array}
      224    0.006    0.000    0.262    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
      224    0.001    0.000    0.255    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
      224    0.002    0.000    0.253    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
      224    0.002    0.000    0.164    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3195(new)
      224    0.160    0.001    0.160    0.001 {built-in method PIL._imaging.fill}
       64    0.001    0.000    0.147    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:249(pad)
      608    0.145    0.000    0.145    0.000 {method 'astype' of 'numpy.ndarray' objects}


=== Qwen2.5-VL fast (W2) ===
output shape:    (175232, 1176)
output bytes:    824.29 MB
peak RSS delta:  3091.14 MB
peak / output:   3.75x

         6259 function calls (6227 primitive calls) in 1.170 seconds

   Ordered by: cumulative time
   List reduced from 138 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.170    1.170 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:87(<lambda>)
        1    0.000    0.000    1.170    1.170 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    1.170    1.170 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:134(preprocess)
        1    0.000    0.000    1.170    1.170 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.146    0.146    1.170    1.170 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:143(_preprocess_image_like_inputs)
        1    0.035    0.035    0.974    0.974 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:182(_preprocess)
        1    0.000    0.000    0.284    0.284 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.284    0.284 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.284    0.284 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.284    0.284 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.283    0.283    0.283    0.283 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        2    0.216    0.108    0.216    0.108 {built-in method torch.cat}
        3    0.177    0.059    0.177    0.059 {method 'reshape' of 'torch._C.TensorBase' objects}
        1    0.029    0.029    0.147    0.147 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
        1    0.000    0.000    0.067    0.067 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
        1    0.000    0.000    0.067    0.067 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)
        1    0.000    0.000    0.067    0.067 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
        2    0.000    0.000    0.059    0.029 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
        2    0.059    0.029    0.059    0.029 {built-in method torch.stack}
        1    0.056    0.056    0.056    0.056 {method 'repeat' of 'torch._C.TensorBase' objects}


=== Qwen2.5-VL legacy (W2) ===
output shape:    (175232, 1176)
output bytes:    824.29 MB
peak RSS delta:  1661.95 MB
peak / output:   2.02x

         14902 function calls (14838 primitive calls) in 2.599 seconds

   Ordered by: cumulative time
   List reduced from 117 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.599    2.599 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:81(<lambda>)
        1    0.007    0.007    2.599    2.599 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.008    0.008    2.592    2.592 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:299(preprocess)
       32    0.158    0.005    2.269    0.071 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:165(_preprocess)
       32    0.001    0.000    0.688    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.000    0.000    0.590    0.018 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.589    0.018    0.589    0.018 {method 'resize' of 'ImagingCore' objects}
      162    0.432    0.003    0.512    0.003 {built-in method numpy.array}
       64    0.474    0.007    0.474    0.007 {method 'reshape' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.457    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.455    0.014    0.457    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
       32    0.000    0.000    0.244    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.071    0.002    0.243    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       96    0.185    0.002    0.185    0.002 {method 'astype' of 'numpy.ndarray' objects}
       64    0.001    0.000    0.079    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.003    0.000    0.078    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:423(repeat)
       32    0.000    0.000    0.075    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
       32    0.075    0.002    0.075    0.002 {method 'repeat' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.062    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
```

# Workload 3

```text
(cs348k-py3.12) jdarve@wheat-01:~/cs348k$ cat profiles/W3/*
=== InternVL2.5 HF Fast (W3) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  165.36 MB
peak / output:   2.15x

         7083 function calls (7051 primitive calls) in 0.235 seconds

   Ordered by: cumulative time
   List reduced from 133 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.235    0.235 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:99(<lambda>)
        1    0.000    0.000    0.235    0.235 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.235    0.235 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.009    0.009    0.234    0.234 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.000    0.000    0.196    0.196 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:781(_preprocess)
       32    0.000    0.000    0.139    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
       32    0.000    0.000    0.138    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       32    0.001    0.000    0.137    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       32    0.000    0.000    0.135    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       32    0.134    0.004    0.134    0.004 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       59    0.038    0.001    0.038    0.001 {built-in method torch.stack}
        1    0.000    0.000    0.029    0.029 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.029    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
       32    0.000    0.000    0.028    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
       32    0.006    0.000    0.026    0.001 {built-in method numpy.array}
        2    0.000    0.000    0.022    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
       32    0.000    0.000    0.020    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       32    0.001    0.000    0.019    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       26    0.000    0.000    0.017    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
       26    0.000    0.000    0.014    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)


=== InternVL2.5 HF Legacy (W3) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak RSS delta:  77.08 MB
peak / output:   1.00x

         11009 function calls (10944 primitive calls) in 0.585 seconds

   Ordered by: cumulative time
   List reduced from 151 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.001    0.000    0.585    0.585 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:93(<lambda>)
        1    0.000    0.000    0.585    0.585 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.000    0.000    0.496    0.496 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:202(preprocess)
       32    0.009    0.000    0.363    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:153(resize)
       32    0.001    0.000    0.353    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.000    0.000    0.304    0.009 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.303    0.009    0.303    0.009 {method 'resize' of 'ImagingCore' objects}
       32    0.000    0.000    0.110    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.109    0.003    0.110    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
      128    0.012    0.000    0.047    0.000 {built-in method numpy.array}
       32    0.000    0.000    0.046    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.014    0.000    0.045    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       96    0.039    0.000    0.039    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.037    0.037 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/feature_extraction_utils.py:77(__init__)
        1    0.000    0.000    0.037    0.037 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/feature_extraction_utils.py:171(convert_to_tensors)
        1    0.000    0.000    0.037    0.037 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/feature_extraction_utils.py:160(as_tensor)
        1    0.036    0.036    0.036    0.036 {built-in method numpy.asarray}
       64    0.000    0.000    0.036    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.002    0.000    0.035    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.028    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)


=== InternVL2.5 Manual Card (W3) ===
output shape:    (196, 3, 448, 448)
output bytes:    472.06 MB
peak RSS delta:  575.26 MB
peak / output:   1.22x

         47072 function calls in 1.621 seconds

   Ordered by: cumulative time
   List reduced from 90 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.011    0.011    1.621    1.621 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:105(<lambda>)
        1    0.002    0.002    1.609    1.609 /home/users/jdarve/cs348k/benchmarks/models.py:104(process_batch)
       32    0.004    0.000    1.178    0.037 /home/users/jdarve/cs348k/benchmarks/models.py:59(dynamic_preprocess)
       63    0.001    0.000    1.136    0.018 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       63    1.134    0.018    1.134    0.018 {method 'resize' of 'ImagingCore' objects}
      196    0.002    0.000    0.284    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:93(__call__)
      196    0.001    0.000    0.193    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:129(__call__)
      196    0.006    0.000    0.193    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:127(to_tensor)
      196    0.020    0.000    0.094    0.000 {built-in method numpy.array}
      392    0.002    0.000    0.088    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1775(_wrapped_call_impl)
      392    0.002    0.000    0.086    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1783(_call_impl)
        1    0.081    0.081    0.081    0.081 {built-in method torch.cat}
      196    0.001    0.000    0.073    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
      196    0.001    0.000    0.073    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:277(forward)
      196    0.002    0.000    0.073    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:327(normalize)
      196    0.005    0.000    0.071    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
      196    0.009    0.000    0.069    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py:905(normalize)
       32    0.064    0.002    0.064    0.002 {built-in method torch.stack}
     1960    0.047    0.000    0.047    0.000 {method 'encode' of 'ImagingEncoder' objects}
      196    0.035    0.000    0.035    0.000 {method 'contiguous' of 'torch._C.TensorBase' objects}


=== LLaVA fast (W3) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  424.78 MB
peak / output:   1.96x

         12447 function calls (12414 primitive calls) in 0.309 seconds

   Ordered by: cumulative time
   List reduced from 194 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.018    0.009    0.290    0.290 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:117(<lambda>)
        1    0.000    0.000    0.290    0.290 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.290    0.290 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:79(preprocess)
        1    0.000    0.000    0.290    0.290 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.000    0.000    0.266    0.266 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.023    0.023    0.240    0.240 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:203(_preprocess)
       32    0.000    0.000    0.130    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:137(_get_image_patches)
       96    0.000    0.000    0.119    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       96    0.002    0.000    0.117    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       64    0.001    0.000    0.112    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       64    0.111    0.002    0.111    0.002 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       64    0.000    0.000    0.071    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
       32    0.000    0.000    0.070    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:83(_resize_for_patching)
       65    0.050    0.001    0.050    0.001 {built-in method torch.stack}
       64    0.000    0.000    0.042    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:5349(pad)
       64    0.041    0.001    0.041    0.001 {built-in method torch._C._nn.pad}
        1    0.000    0.000    0.036    0.036 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:181(_pad_for_batching)
        1    0.000    0.000    0.024    0.024 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.024    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
       32    0.000    0.000    0.024    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)


=== LLaVA legacy (W3) ===
output shape:    (32, 5, 3, 336, 336)
output bytes:    216.76 MB
peak RSS delta:  352.11 MB
peak / output:   1.62x

         51332 function calls (51268 primitive calls) in 1.308 seconds

   Ordered by: cumulative time
   List reduced from 160 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.308    1.308 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:111(<lambda>)
        1    0.004    0.004    1.308    1.308 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.001    0.001    1.304    1.304 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:550(preprocess)
      205    0.003    0.000    0.681    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.001    0.000    0.616    0.019 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:447(get_image_patches)
      205    0.002    0.000    0.524    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       64    0.516    0.008    0.516    0.008 {method 'resize' of 'ImagingCore' objects}
       32    0.002    0.000    0.460    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:317(_preprocess)
       32    0.000    0.000    0.349    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:401(_resize_for_patching)
      141    0.001    0.000    0.271    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
      141    0.267    0.002    0.270    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
      775    0.076    0.000    0.141    0.000 {built-in method numpy.array}
       64    0.000    0.000    0.097    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:249(pad)
      141    0.000    0.000    0.097    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
      141    0.031    0.000    0.096    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       64    0.002    0.000    0.096    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/lib/arraypad.py:533(pad)
      205    0.002    0.000    0.089    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
      551    0.086    0.000    0.086    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.085    0.085 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:511(_pad_for_batching)
      141    0.001    0.000    0.082    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:199(resize)


=== Qwen2.5-VL fast (W3) ===
output shape:    (70880, 1176)
output bytes:    333.42 MB
peak RSS delta:  773.30 MB
peak / output:   2.32x

         7422 function calls (7390 primitive calls) in 0.478 seconds

   Ordered by: cumulative time
   List reduced from 138 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.478    0.478 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:87(<lambda>)
        1    0.000    0.000    0.478    0.478 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.478    0.478 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:134(preprocess)
        1    0.000    0.000    0.478    0.478 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.054    0.054    0.477    0.477 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:143(_preprocess_image_like_inputs)
        1    0.003    0.003    0.395    0.395 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:182(_preprocess)
       32    0.000    0.000    0.156    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
       32    0.000    0.000    0.156    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       32    0.001    0.000    0.155    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       32    0.000    0.000    0.153    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       32    0.152    0.005    0.152    0.005 {built-in method torch._C._nn._upsample_bicubic2d_aa}
       32    0.097    0.003    0.097    0.003 {built-in method torch.cat}
       95    0.068    0.001    0.068    0.001 {method 'reshape' of 'torch._C.TensorBase' objects}
       31    0.001    0.000    0.033    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
        2    0.000    0.000    0.029    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
       63    0.028    0.000    0.028    0.000 {built-in method torch.stack}
        1    0.000    0.000    0.028    0.028 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.028    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
       32    0.000    0.000    0.027    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
       32    0.006    0.000    0.025    0.001 {built-in method numpy.array}


=== Qwen2.5-VL legacy (W3) ===
output shape:    (70880, 1176)
output bytes:    333.42 MB
peak RSS delta:  418.19 MB
peak / output:   1.25x

         11208 function calls (11144 primitive calls) in 1.365 seconds

   Ordered by: cumulative time
   List reduced from 117 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.365    1.365 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:81(<lambda>)
        1    0.000    0.000    1.365    1.365 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.006    0.006    1.365    1.365 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:299(preprocess)
       32    0.072    0.002    1.204    0.038 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:165(_preprocess)
       32    0.001    0.000    0.415    0.013 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
       32    0.000    0.000    0.358    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       32    0.357    0.011    0.357    0.011 {method 'resize' of 'ImagingCore' objects}
       32    0.000    0.000    0.267    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
       32    0.266    0.008    0.267    0.008 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
      162    0.204    0.001    0.250    0.002 {built-in method numpy.array}
       64    0.224    0.003    0.224    0.003 {method 'reshape' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.113    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
       32    0.033    0.001    0.113    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       96    0.084    0.001    0.084    0.001 {method 'astype' of 'numpy.ndarray' objects}
       64    0.001    0.000    0.046    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       64    0.003    0.000    0.044    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.032    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
       32    0.000    0.000    0.030    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:423(repeat)
       32    0.000    0.000    0.029    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
       32    0.029    0.001    0.029    0.001 {method 'repeat' of 'numpy.ndarray' objects}
```

# Workload 4

```text
(cs348k-py3.12) jdarve@wheat-01:~/cs348k$ cat profiles/W4/*
=== InternVL2.5 HF Fast (W4) ===
output shape:    (8, 3, 448, 448)
output bytes:    19.27 MB
peak RSS delta:  365.27 MB
peak / output:   18.96x

         7967 function calls (7959 primitive calls) in 0.462 seconds

   Ordered by: cumulative time
   List reduced from 133 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.462    0.462 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:99(<lambda>)
        1    0.000    0.000    0.462    0.462 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.002    0.002    0.462    0.462 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.000    0.000    0.460    0.460 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.021    0.021    0.299    0.299 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:781(_preprocess)
        1    0.000    0.000    0.212    0.212 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.212    0.212 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.212    0.212 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.212    0.212 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.212    0.212    0.212    0.212 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        1    0.000    0.000    0.161    0.161 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
        8    0.000    0.000    0.161    0.020 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
        8    0.000    0.000    0.160    0.020 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
        8    0.068    0.008    0.159    0.020 {built-in method numpy.array}
        8    0.000    0.000    0.092    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
        8    0.002    0.000    0.091    0.011 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
        3    0.063    0.021    0.063    0.021 {built-in method torch.stack}
        2    0.000    0.000    0.063    0.031 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:886(group_images_by_shape)
     3512    0.051    0.000    0.051    0.000 {method 'encode' of 'ImagingEncoder' objects}
        8    0.038    0.005    0.038    0.005 {method 'join' of 'bytes' objects}


=== InternVL2.5 HF Legacy (W4) ===
output shape:    (8, 3, 448, 448)
output bytes:    19.27 MB
peak RSS delta:  225.44 MB
peak / output:   11.70x

         9458 function calls (9442 primitive calls) in 0.922 seconds

   Ordered by: cumulative time
   List reduced from 116 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.922    0.922 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:93(<lambda>)
        1    0.000    0.000    0.922    0.922 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.004    0.004    0.922    0.922 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:202(preprocess)
        8    0.000    0.000    0.709    0.089 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/clip/image_processing_clip.py:153(resize)
        8    0.000    0.000    0.709    0.089 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
        8    0.000    0.000    0.568    0.071 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
        8    0.568    0.071    0.568    0.071 {method 'resize' of 'ImagingCore' objects}
       32    0.079    0.002    0.176    0.006 {built-in method numpy.array}
        8    0.000    0.000    0.171    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
        8    0.000    0.000    0.136    0.017 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
       16    0.000    0.000    0.097    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
        8    0.000    0.000    0.096    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
       16    0.002    0.000    0.096    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
        8    0.000    0.000    0.096    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
        8    0.000    0.000    0.096    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
     3624    0.051    0.000    0.051    0.000 {method 'encode' of 'ImagingEncoder' objects}
        8    0.000    0.000    0.050    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3195(new)
        8    0.050    0.006    0.050    0.006 {built-in method PIL._imaging.fill}
       24    0.046    0.002    0.046    0.002 {method 'astype' of 'numpy.ndarray' objects}
        8    0.000    0.000    0.046    0.006 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:932(frombytes)


=== InternVL2.5 Manual Card (W4) ===
output shape:    (56, 3, 448, 448)
output bytes:    134.87 MB
peak RSS delta:  216.72 MB
peak / output:   1.61x

         13052 function calls in 1.431 seconds

   Ordered by: cumulative time
   List reduced from 90 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.008    0.008    1.431    1.431 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:105(<lambda>)
        1    0.000    0.000    1.422    1.422 /home/users/jdarve/cs348k/benchmarks/models.py:104(process_batch)
        8    0.001    0.000    1.310    0.164 /home/users/jdarve/cs348k/benchmarks/models.py:59(dynamic_preprocess)
       16    0.000    0.000    1.300    0.081 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       16    1.299    0.081    1.299    0.081 {method 'resize' of 'ImagingCore' objects}
       56    0.000    0.000    0.078    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:93(__call__)
       56    0.000    0.000    0.053    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:129(__call__)
       56    0.001    0.000    0.053    0.001 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:127(to_tensor)
      112    0.000    0.000    0.024    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1775(_wrapped_call_impl)
      112    0.000    0.000    0.024    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/modules/module.py:1783(_call_impl)
       56    0.005    0.000    0.022    0.000 {built-in method numpy.array}
        1    0.021    0.021    0.021    0.021 {built-in method torch.cat}
       56    0.000    0.000    0.021    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/transforms.py:277(forward)
       56    0.000    0.000    0.021    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:327(normalize)
       56    0.002    0.000    0.020    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py:905(normalize)
       56    0.000    0.000    0.017    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       56    0.001    0.000    0.016    0.000 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       56    0.015    0.000    0.015    0.000 {method 'contiguous' of 'torch._C.TensorBase' objects}
        8    0.013    0.002    0.013    0.002 {built-in method torch.stack}
      560    0.011    0.000    0.011    0.000 {method 'encode' of 'ImagingEncoder' objects}


=== LLaVA fast (W4) ===
output shape:    (8, 5, 3, 336, 336)
output bytes:    54.19 MB
peak RSS delta:  283.87 MB
peak / output:   5.24x

         9909 function calls (9901 primitive calls) in 0.431 seconds

   Ordered by: cumulative time
   List reduced from 159 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.431    0.431 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:117(<lambda>)
        1    0.000    0.000    0.431    0.431 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.431    0.431 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:79(preprocess)
        1    0.008    0.008    0.431    0.431 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.000    0.000    0.423    0.423 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.000    0.000    0.253    0.253 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:203(_preprocess)
        8    0.000    0.000    0.225    0.028 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:137(_get_image_patches)
       24    0.000    0.000    0.221    0.009 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
       24    0.001    0.000    0.221    0.009 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
       16    0.000    0.000    0.219    0.014 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
       16    0.218    0.014    0.218    0.014 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        1    0.000    0.000    0.170    0.170 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
        8    0.000    0.000    0.170    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
        8    0.000    0.000    0.169    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
        8    0.074    0.009    0.168    0.021 {built-in method numpy.array}
       16    0.000    0.000    0.118    0.007 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        8    0.000    0.000    0.118    0.015 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next_fast.py:83(_resize_for_patching)
        8    0.000    0.000    0.094    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
        8    0.002    0.000    0.094    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
     3512    0.051    0.000    0.051    0.000 {method 'encode' of 'ImagingEncoder' objects}


=== LLaVA legacy (W4) ===
output shape:    (8, 5, 3, 336, 336)
output bytes:    54.19 MB
peak RSS delta:  303.44 MB
peak / output:   5.60x

         20663 function calls (20647 primitive calls) in 1.615 seconds

   Ordered by: cumulative time
   List reduced from 160 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.615    1.615 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:111(<lambda>)
        1    0.006    0.006    1.615    1.615 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.000    0.000    1.609    1.609 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:550(preprocess)
       56    0.001    0.000    1.311    0.023 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
        8    0.000    0.000    1.298    0.162 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:447(get_image_patches)
       56    0.000    0.000    1.017    0.018 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
       16    1.014    0.063    1.014    0.063 {method 'resize' of 'ImagingCore' objects}
        8    0.000    0.000    0.681    0.085 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:401(_resize_for_patching)
       56    0.000    0.000    0.278    0.005 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:162(to_pil_image)
       56    0.001    0.000    0.197    0.004 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3374(fromarray)
       56    0.000    0.000    0.196    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3290(frombuffer)
      208    0.094    0.000    0.196    0.001 {built-in method numpy.array}
       56    0.000    0.000    0.195    0.003 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3244(frombytes)
        8    0.000    0.000    0.169    0.021 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_utils.py:287(to_numpy_array)
       64    0.001    0.000    0.102    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       56    0.000    0.000    0.101    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:3195(new)
       64    0.003    0.000    0.101    0.002 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
       56    0.101    0.002    0.101    0.002 {built-in method PIL._imaging.fill}
        8    0.000    0.000    0.100    0.012 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py:317(_preprocess)
      152    0.096    0.001    0.096    0.001 {method 'astype' of 'numpy.ndarray' objects}


=== Qwen2.5-VL fast (W4) ===
output shape:    (356000, 1176)
output bytes:    1674.62 MB
peak RSS delta:  6436.12 MB
peak / output:   3.84x

         7987 function calls (7979 primitive calls) in 2.798 seconds

   Ordered by: cumulative time
   List reduced from 138 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.798    2.798 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:87(<lambda>)
        1    0.000    0.000    2.798    2.798 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    2.798    2.798 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:134(preprocess)
        1    0.002    0.002    2.798    2.798 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:734(preprocess)
        1    0.331    0.331    2.797    2.797 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:143(_preprocess_image_like_inputs)
        1    0.080    0.080    2.304    2.304 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py:182(_preprocess)
        1    0.000    0.000    0.793    0.793 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.793    0.793 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.793    0.793 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.793    0.793 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torch/nn/functional.py:4614(interpolate)
        1    0.793    0.793    0.793    0.793 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        2    0.470    0.235    0.470    0.235 {built-in method torch.cat}
        3    0.405    0.135    0.405    0.135 {method 'reshape' of 'torch._C.TensorBase' objects}
        1    0.067    0.067    0.327    0.327 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:450(rescale_and_normalize)
        1    0.000    0.000    0.161    0.161 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:605(_prepare_image_like_inputs)
        8    0.000    0.000    0.161    0.020 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:567(_process_image)
        8    0.000    0.000    0.161    0.020 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/functional.py:181(pil_to_tensor)
        8    0.068    0.008    0.160    0.020 {built-in method numpy.array}
        1    0.000    0.000    0.143    0.143 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils_fast.py:410(normalize)
        1    0.000    0.000    0.143    0.143 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/torchvision/transforms/v2/functional/_misc.py:19(normalize)


=== Qwen2.5-VL legacy (W4) ===
output shape:    (356000, 1176)
output bytes:    1674.62 MB
peak RSS delta:  3382.93 MB
peak / output:   2.02x

         16214 function calls (16198 primitive calls) in 7.283 seconds

   Ordered by: cumulative time
   List reduced from 117 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    7.283    7.283 /home/users/jdarve/cs348k/benchmarks/full_benchmark.py:81(<lambda>)
        1    0.001    0.001    7.283    7.283 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:49(__call__)
        1    0.042    0.042    7.282    7.282 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:299(preprocess)
        8    0.607    0.076    6.579    0.822 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:165(_preprocess)
        8    0.000    0.000    1.805    0.226 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:323(resize)
        8    0.000    0.000    1.410    0.176 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:2328(resize)
        8    1.409    0.176    1.409    0.176 {method 'resize' of 'ImagingCore' objects}
       42    1.113    0.026    1.405    0.033 {built-in method numpy.array}
       16    1.325    0.083    1.325    0.083 {method 'reshape' of 'numpy.ndarray' objects}
        8    0.000    0.000    1.173    0.147 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:88(normalize)
        8    1.172    0.147    1.172    0.147 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:394(normalize)
        8    0.000    0.000    0.860    0.107 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_processing_utils.py:56(rescale)
        8    0.160    0.020    0.860    0.107 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/transformers/image_transforms.py:97(rescale)
       24    0.732    0.031    0.732    0.031 {method 'astype' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.292    0.018 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:811(__array_interface__)
       16    0.004    0.000    0.291    0.018 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/PIL/Image.py:849(tobytes)
        8    0.000    0.000    0.286    0.036 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:423(repeat)
        8    0.000    0.000    0.286    0.036 /home/users/jdarve/.cache/pypoetry/virtualenvs/cs348k-CgSSqESF-py3.12/lib/python3.12/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
        8    0.286    0.036    0.286    0.036 {method 'repeat' of 'numpy.ndarray' objects}
       16    0.187    0.012    0.187    0.012 {method 'join' of 'bytes' objects}

```