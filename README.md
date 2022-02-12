# python-machine-learning-class
Machine Learning Class Implementation with Python

## Reference
Github: https://github.com/bjpublic/MLpythonpiece
Book: http://www.yes24.com/Product/Goods/87663679

## Example

```pyhon
x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10,1)
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10,1)

model = BinaryClassification(x_data, t_data)
model.train(50001, 1e-2)
result1 = model.predict(np.array([3.0]))
result2 = model.predict(np.array([17.0]))

print('3시간 공부하면', '합격' if result1 >= 0.5 else '불합격')
print('17시간 공부하면', '합격' if result2 >= 0.5 else '불합격')
```

```
step =  0 loss value =  18.926494323909623
step =  5000 loss value =  0.8437482116252524
step =  10000 loss value =  0.6203878952312823
step =  15000 loss value =  0.5091379860919384
step =  20000 loss value =  0.43753823488841864
step =  25000 loss value =  0.3859665387608092
step =  30000 loss value =  0.34638422590689105
step =  35000 loss value =  0.31473548087786885
step =  40000 loss value =  0.2886945306848048
step =  45000 loss value =  0.26680660036735715
step =  50000 loss value =  0.24810279323523302
3시간 공부하면 불합격
17시간 공부하면 합격
```

[Google Colab Link](https://colab.research.google.com/drive/1-CcjJRt1IKRN_y98ns-eFKase_7Aqs1-?usp=sharing)
