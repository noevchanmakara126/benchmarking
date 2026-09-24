numbers_list =[10]

def find_missing1(numbers):
 n = len(numbers)
 return n * (n + 1) // 2 - sum(numbers)


def find_missing2(numbers):
 n = len(numbers)
 expected_sum = n * (n + 1) // 2
 actual_sum = sum(numbers)
 return expected_sum - actual_sum

result1 = find_missing1(numbers_list)
result2 = find_missing2(numbers_list)

print("Missing number on result 1 :",result1)
print("Missing number on result 2 :",result2)