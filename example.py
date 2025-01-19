import sys
sys.path.append('/home/adanilishin/myTorch/build')
import ctypes
ctypes.CDLL('/home/adanilishin/myTorch/build/libtorchcpp.cpython-312-aarch64-linux-gnu.so')
import libtorchcpp

# # Создание тензора
# tensor1 = libtorchcpp.Tensor([2, 3])  # 2x3 тензор
# tensor2 = libtorchcpp.Tensor([2, 3])  # 2x3 тензор

# Заполнение тензоров
tensor1 = libtorchcpp.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor2.data = libtorchcpp.Tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])

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