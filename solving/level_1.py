'''
배열 array의 i번째 숫자부터 j번째 숫자까지 자르고 정렬했을 때,
k번째에 있는 수를 구하려 합니다.예를 들어 array가[1, 5, 2, 6, 3, 7, 4],
i = 2,
j = 5,
k = 3 이라면

array의 2 번째부터 5 번째까지 자르면[5, 2, 6, 3] 입니다.1 에서 나온 배열을 정렬하면[2, 3, 5, 6] 입니다.
2 에서 나온 배열의 3 번째 숫자는 5 입니다.배열 array,
[i, j, k] 를 원소로 가진 2 차원 배열 commands가 매개변수로 주어질 때,
commands의 모든 원소에 대해 앞서 설명한 연산을 적용했을 때 나온 결과를 배열에 담아 
return 하도록 solution 함수를 작성해주세요.제한사항
array의 길이는 1 이상 100 이하입니다.array의 각 원소는 1 이상 100 이하입니다.
commands의 길이는 1 이상 50 이하입니다.commands의 각 원소는 길이가 3 입니다.
'''

from tkinter import E


array = [1, 5, 2, 6, 3, 7, 4]
commands = [[2, 5, 3], [4, 4, 1], [1, 7, 3]]


# def solution1(array, commands):
#     return list(map(lambda x: sorted(array[x[0]-1:x[1]])[x[2]-1], commands))

# print(solution1())

# def solution2(array, commands):
#     answer = []
#     for command in commands:
#         i, j, k = command
#         answer.append(list(sorted(array[i-1:j]))[k-1])
#     return answer


# def solution3(array, commands):
#     result = []
#     for command in commands:
#         i, j, k = command[0], command[1], command[2]
#         subarray = sorted(array[i-1:j])
#         result.append(subarray[k-1])
#     return result


# def solution4(array, commands):
#     answer = []
#     for it in commands:
#         [i, j, k] = it
#         temp = array[i-1:j]
#         temp.sort()
#         answer.append(temp[k-1])
#     return answer


