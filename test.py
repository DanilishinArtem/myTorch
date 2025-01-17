import sys
sys.path.append('/home/adanilishin/myTorch/build')
import ctypes
ctypes.CDLL('/home/adanilishin/myTorch/build/libtorchcpp.cpython-312-x86_64-linux-gnu.so')
import libtorchcpp

# first type of initialization
# tensor1 = libtorchcpp.Tensor([2, 3])
# tensor2 = libtorchcpp.Tensor([2, 3])
# tensor1.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# tensor2.data = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

# second type of iniziatization
tensor1 = libtorchcpp.Tensor([[[1.1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
tensor2 = libtorchcpp.Tensor([[[1.1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])

# Операции
result_add = tensor1.add(tensor2)
result_mult = tensor1.multiply(tensor2)

result_add[0,0] = 100
result_mult[0,0] = -100

print(result_add[0,0])
# # Вывод результата
print("Sum result:")
result_add.print()

print("Multiply result:")
result_mult.print()